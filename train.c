#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <ctype.h>
#include <errno.h>
#include <mpi.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	if(mpi_get_rank() != 0)
		mpi_exit(1);
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 0)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"	
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v type: cross validation by different ways, type is 1 or 2\n"
	"-C type: find the best C by different ways of cross validation, type is 1 or 2\n"
	"-q : quiet mode (no outputs)\n"
	);
	mpi_exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void get_output_file_name(const char* input_file_name, char* output_file_name);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
int flag_find_C;
int C_type;
int flag_cross_validation;
int v_type;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);	

	char input_file_name[1024];
	char output_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;	

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	get_output_file_name(input_file_name, output_file_name);
	error_msg = check_parameter(&prob,&param);	
	
	mpi_allreduce(&prob.n, 1, MPI_INT, MPI_MAX);	
	
	if(error_msg)
	{
		if(mpi_get_rank()==0)
			fprintf(stderr,"ERROR: %s\n", error_msg);
		mpi_exit(1);
	}

	else if(flag_find_C) // find best C
	{
		if(mpi_get_rank() == 0)
			printf("Find parameter C%d for %s\n", C_type, input_file_name);
		double best_C, best_rate;
		double max_C = 1024;	
		double start_C = get_startC();
		FILE* fp = NULL;
		char* cursor = output_file_name;
		while(*cursor != '\0')
			cursor++;
		if(C_type == 1)
		{
			strcpy(cursor, ".output.1");	// output file name is home/jing/outputs/ijcnn.output.1
			fp = fopen(output_file_name, "w");
			if(fp == NULL)
			{
				printf("%d can not open file %s\n", mpi_get_rank(), output_file_name);	
				mpi_exit(1);
			}				
			else
				find_parameter_C1(&prob, &param, start_C, max_C, &best_C, &best_rate, fp);	// linear.cpp			
		}
		else
		{
			strcpy(cursor, ".output.2");	// output file name is home/jing/outputs/ijcnn.output.2
			fp = fopen(output_file_name, "w");
			if(fp == NULL)
			{
				if(mpi_get_rank() == 0)
					printf("can not open file %s\n", output_file_name);
			}			
			else
				find_parameter_C2(&prob, &param, start_C, max_C, &best_C, &best_rate, fp);	// linear.cpp
		}
		if(mpi_get_rank() == 0)
			printf("log2(best_C) = %g, best_rate = %g%%\n", log(best_C) / log(2.0), 100 * best_rate);
		if(fp != NULL)
		{
			fprintf(fp, "log2(best_C) = %g, best_rate = %g%%\n", log(best_C) / log(2.0), 100 * best_rate);
			fclose(fp);
		}
	}
	
	else if(flag_cross_validation) // do cross validation
	{
		double accuracy;
		if(v_type == 1)
			accuracy = cross_validation1(&prob, &param);
		else
			accuracy = cross_validation2(&prob, &param);
		if(mpi_get_rank() == 0)
			printf("CV accurracy is %g%%\n", accuracy * 100);
	}
	
	else
	{
		double *init_w = Malloc(double, prob.n);
		for(int i = 0; i < prob.n; i++)
			init_w[i] = 0;
		model *model_=train(&prob, &param, init_w, -1);
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"[rank %d] can't save model to file %s\n", mpi_get_rank(), model_file_name);
			mpi_exit(1);
		}
		free(init_w);
		free_and_destroy_model(&model_);
	}
	
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	MPI_Finalize();
	return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_LR;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 0;
	flag_find_C = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = L2R_LR;
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;			
				v_type = atoi(argv[i]);
				break;

			case 'C':
				flag_find_C = 1;
				C_type = atoi(argv[i]);
				break;
				
			case 'q':
				print_func = &print_null;
				i--;
				break;
			
			default:
				if(mpi_get_rank() == 0)
					fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(param.eps == INF)
		param.eps = 0.01;
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n",mpi_get_rank(),filename);
		mpi_exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}

// input_file_name is like /home/jing/dis_data/ijcnn.sub.1
// output_file_name should be like /home/jing/outputs/ijcnn
void get_output_file_name(const char* input_file_name, char* output_file_name)
{	
	const char* input_directory = "/home/jing/dis_data/";
	const char* output_directory = "/home/jing/outputs/";
	strcpy(output_file_name, output_directory);
	size_t i, o;
	for(i = strlen(input_directory), o = strlen(output_directory); input_file_name[i] != '.'; i++, o++)
		output_file_name[o] = input_file_name[i];
	output_file_name[o] = '\0';
}