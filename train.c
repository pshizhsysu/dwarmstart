#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <mpi.h>
#include "structures.h"
#include "linear.h"
#include "mpi_fun.h"

static feature_node *x_space;
static parameter param;
static data data_obj;
static int flag_find_C;
static int flag_cross_validation;
static int nr_fold;		// for cross_validation and find_parameter_C
static char *line;
static int max_line_len;

static void do_find_parameter_C();
static void do_cross_validation();
static void initialize_global_variables();
static void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
static char* readline(FILE *input);
static void read_data(const char *filename);
static const char *check_parameter(const parameter *param_ptr);
static void exit_with_help();
static void exit_input_error(int line_num);

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);	

	char input_file_name[1024];	
	char model_file_name[1024];
	const char *error_msg;	
	initialize_global_variables();
	parse_command_line(argc, argv, input_file_name, model_file_name);	
	if(mpi_get_rank() == 0)		
		fprintf(stderr, "-------------------- %s, data loading ...........\n", input_file_name);
	read_data(input_file_name);	
	error_msg = check_parameter(&param);	
	if(mpi_get_size() != 1)
		mpi_allreduce(&data_obj.n, 1, MPI_INT, MPI_MAX);		
	if(error_msg)
	{
		if(mpi_get_rank() == 0)
			fprintf(stderr,"ERROR: %s\n", error_msg);
		mpi_exit(1);
	}	
	
	if(flag_find_C) // find best C
	{
		do_find_parameter_C();		
	}	
	else if(flag_cross_validation) // do cross validation
	{
		do_cross_validation();		
	}	
	else
	{
		model *model_ptr = malloc_model(data_obj.n);	
		
		train(&data_obj, &param, model_ptr);
		double test_accuracy = test(model_ptr, &data_obj);
		if(mpi_get_rank() == 0)
			fprintf(stderr, "%.6f\n", test_accuracy);
		
	/*	if(*model_file_name != '\0')
		{
			if(save_model(model_file_name, model_ptr) != 0)
			{
				fprintf(stderr, "[%d] can't save model to file %s\n", mpi_get_rank(), model_file_name);
				mpi_exit(1);
			}
		}
	*/
		free_model(model_ptr);
	}	

	free(data_obj.y);
	free(data_obj.x);
	free(x_space);
	free(line);

	MPI_Finalize();
	return 0;
}

static void do_find_parameter_C()
{
	double start_C, best_C, best_rate, last_C;
	double max_C = 1024;	
	start_C = -1;
	find_parameter_C(&data_obj, &param, nr_fold, start_C, max_C, &best_C, &best_rate, &last_C);	// linear.cpp		
	if(mpi_get_rank() == 0)
	{
		fprintf(stderr, "best_C = %.2f best_rate = %.6f last_C = %.2f\n", log(best_C) / log(2.0), 100 * best_rate, log(last_C) / log(2.0));
	}		
}

static void do_cross_validation()
{
	double start = MPI_Wtime();
	double accuracy = cross_validation(&data_obj, &param, nr_fold);
	double end = MPI_Wtime();
	if(mpi_get_rank() == 0)
	{		
		fprintf(stderr, "%.2f %g%%\n", end - start, accuracy * 100);				
	}

}

static void initialize_global_variables()
{	
	// other global varibales are set to 0 or NULL by default
	param.C = 1;
	param.eps = 0.01;
	param.eps_cg = 0.1;	
	nr_fold = 5;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{	
	// parse options
	int i;
	for(i = 1; i < argc; i++)
	{
		if(argv[i][0] != '-') break;
		if(++i >= argc)
			exit_with_help();
		switch(argv[i - 1][1])
		{
			case 'c':
				param.C = atof(argv[i]);				
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'g':
				param.eps_cg = atof(argv[i]);
				break;
			case 'v':
				flag_cross_validation = 1;			
				nr_fold = atoi(argv[i]);
				break;
			case 'C':
				flag_find_C = 1;
				i--;
				break;			
			default:
				if(mpi_get_rank() == 0)
					fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}	

	// determine filenames
	if(i >= argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i < argc - 1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		// if model_file_name is not specified, do not save model to file
		*model_file_name = '\0';		
	}
}

static char* readline(FILE *input)
{
	int len;

	if(fgets(line, max_line_len, input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

void read_data(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n", mpi_get_rank(), filename);
		mpi_exit(1);
	}

	data_obj.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = (char*)malloc(sizeof(char) * max_line_len);
	while(readline(fp) != NULL)
	{
		char *p = strtok(line, " \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		data_obj.l++;
	}
	rewind(fp);

	data_obj.y = (double*)malloc(sizeof(double) * data_obj.l);
	data_obj.x = (feature_node**)malloc(sizeof(feature_node*) * data_obj.l);
	x_space = (feature_node*)malloc(sizeof(feature_node) * (elements + data_obj.l));

	max_index = 0;
	j = 0;
	for(i = 0; i < data_obj.l; i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		data_obj.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if(label == NULL) // empty line
			exit_input_error(i + 1);

		data_obj.y[i] = strtod(label, &endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i + 1);

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx, &endptr, 10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;	

		x_space[j++].index = -1;
	}

	data_obj.n = max_index;

	fclose(fp);
}

static const char *check_parameter(const parameter *param_ptr)
{
	if(param_ptr->eps <= 0)
		return "eps <= 0";
	
	if(param_ptr->eps <= 0)
		return "eps_cg <= 0";
	
	if(param_ptr->C <= 0)
		return "C <= 0";

	return NULL;
}

static void exit_with_help()
{
	if(mpi_get_rank() != 0)
		mpi_exit(1);
	fprintf(stderr,
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-c cost : set the parameter C (default 1)\n"
	"-e epsilon : set termination criterion for TRON, default 0.01\n"
	"-g epsilon_cg : set termination criterion for CG, default 0.1\n"
	"-v k: k-fold cross validation\n"
	"-C : find the best C\n"
	);
	mpi_exit(1);
}

static void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

