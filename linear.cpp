#include <assert.h>
#include <stdlib.h>
#include <set>
#include <map>
#include <math.h>
#include "structures.h"
#include "linear.h"
#include "tron.h"
#include "mpi_fun.h"

using namespace std;

static double calc_start_C(const data *data_ptr, const parameter *param_ptr);
static void group_classes(const data *data_ptr, int *perm, int *label, int* positive);
static void train_one(const data *data_ptr, const parameter *param_ptr, double *w);

void train(const data *data_ptr, const parameter *param_ptr, model *model_ptr)
{
	int l = data_ptr->l;
	int n = data_ptr->n;	
	int positive;
	int label[2];	
	int *perm = (int*)malloc(sizeof(int) * l);	
	group_classes(data_ptr, perm, label, &positive);	
	model_ptr->label[0] = label[0];
	model_ptr->label[1] = label[1];
	// change label of training data to +1 or -1
	data sub_data;
	sub_data.l = l;
	sub_data.n = n;
	sub_data.x = (feature_node**)malloc(sizeof(feature_node*) * sub_data.l);
	sub_data.y = (double*)malloc(sizeof(double) * sub_data.l);
	for(int i = 0; i < l; i++)
		sub_data.x[i] = data_ptr->x[perm[i]];	
	for(int k = 0; k < positive; k++)
		sub_data.y[k] = +1;
	for(int k = positive; k < sub_data.l; k++)
		sub_data.y[k] = -1;	
	model_ptr->n = n;
	train_one(&sub_data, param_ptr, model_ptr->w);	
	free(perm);
	free(sub_data.x);
	free(sub_data.y);
}

double cross_validation(const data *data_ptr, const parameter *param_ptr, int nr_fold)
{	
	assert(nr_fold == 5);
	int l = data_ptr->l;
	int n = data_ptr->n;
	double* target = (double*)malloc(sizeof(double) * l);	
	
	// constructing sub-data by training data : start
	data *subdata = (data*)malloc(sizeof(data) * nr_fold);
	for(int i = 0; i < nr_fold; i++)
	{		
		subdata[i].n = data_ptr->n;
		subdata[i].l = l - (l / nr_fold + (l % nr_fold > i ? 1 : 0));
		subdata[i].x = (feature_node**)malloc(sizeof(feature_node*) * subdata[i].l);
		subdata[i].y = (double*)malloc(sizeof(double) * subdata[i].l);
	}
	
	for(int i = 0; i < nr_fold; i++)
	{
		int ix = 0;
		for(int j = 0; j < l; j++)
		{
			if(j % nr_fold != i)
			{
				subdata[i].x[ix] = data_ptr->x[j];
				subdata[i].y[ix] = data_ptr->y[j];
				ix++;
			}
		}
	}
	// constructing sub-data by training data : end
	
	// cross validation : start
	for(int i=0;i<nr_fold;i++)
	{
		model *submodel = malloc_model(n);			
		train(&subdata[i],param_ptr,submodel);	
		for(int k = 0; k < l - subdata[i].l; k++)
			target[i + k * nr_fold] = predict(submodel,data_ptr->x[i + k * nr_fold]);
		free_model(submodel);
	}
	int total_correct = 0;
	int global_l = l;
	for(int i = 0; i < l; i++)
		if(target[i] == data_ptr->y[i])
			total_correct++;
	if(mpi_get_size() != 1)
	{
		mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);
		mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
	}
	double cv_accuracy = (double)total_correct / global_l;
		
	free(target);	
	for(int i=0; i<nr_fold; i++)
	{
		free(subdata[i].x);
		free(subdata[i].y);
	}
	free(subdata);
	return cv_accuracy;
}

void find_parameter_C(const data *data_ptr, parameter *param_ptr, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate, double *last_C)		
{
	assert(nr_fold == 5);
	int l = data_ptr->l;
	int n = data_ptr->n;
	double* target = (double*)malloc(sizeof(double) * l);
			
	data *subdata = (data*)malloc(sizeof(data) * nr_fold);
	for(int i = 0; i < nr_fold; i++)
	{		
		subdata[i].n = data_ptr->n;
		subdata[i].l = l - (l / nr_fold + (l % nr_fold > i ? 1 : 0));
		subdata[i].x = (feature_node**)malloc(sizeof(feature_node*) * subdata[i].l);
		subdata[i].y = (double*)malloc(sizeof(double) * subdata[i].l);
	}
	
	for(int i = 0; i < nr_fold; i++)
	{
		int ix = 0;
		for(int j = 0; j < l; j++)
		{
			if(j % nr_fold != i)
			{
				subdata[i].x[ix] = data_ptr->x[j];
				subdata[i].y[ix] = data_ptr->y[j];
				ix++;
			}
		}
	}
	// construting sub-data by training data : end
	
	double ratio = 2;
	int num_unchanged_w = -1;		
	double **w_array = (double**)malloc(sizeof(double*) * nr_fold);
	for(int i = 0; i < nr_fold; i++)
	{	
		w_array[i] = (double*)malloc(sizeof(double) * n);
		for(int j = 0; j < n; j++)
			w_array[i][j] = 0;
	}	
	
	*best_rate = 0.0;
	*last_C = 1024;
	if(start_C <= 0)
		start_C = calc_start_C(data_ptr, param_ptr);	
	
	for(param_ptr->C = start_C; param_ptr->C <= max_C; param_ptr->C *= ratio)
	{	
		double start = MPI_Wtime();
		for(int i=0;i<nr_fold;i++)
		{
			model *submodel = malloc_model(n);
			for(int j = 0; j < n; j++)
				submodel->w[j] = w_array[i][j];
			train(&subdata[i],param_ptr,submodel);	
			for(int k = 0; k < l - subdata[i].l; k++)
				target[i + k * nr_fold] = predict(submodel,data_ptr->x[i + k * nr_fold]);
			
			if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(int j=0; j<n; j++)			
					norm_w_diff += (submodel->w[j] - w_array[i][j])*(submodel->w[j] - w_array[i][j]);				
				norm_w_diff = sqrt(norm_w_diff);
				if(norm_w_diff > 1e-15)
				{
					num_unchanged_w = -1;					
				}
			}
			
			// warm start
			for(int k = 0; k < n; k++)
				w_array[i][k] = submodel->w[k];	
			free_model(submodel);
		}	// for(int i=0;i<nr_fold;i++)
		
		int total_correct = 0;
		int global_l = l;
		for(int i = 0; i < l; i++)
			if(target[i] == data_ptr->y[i])
				total_correct++;
		if(mpi_get_size() != 1)
		{
			mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);
			mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
		}
		double cv_accuracy = (double)total_correct / global_l;		
		if(cv_accuracy > *best_rate)
		{
			*best_rate = cv_accuracy;
			*best_C = param_ptr -> C;
		}
		double end = MPI_Wtime();
		double cv_time = (end - start);
		if(mpi_get_rank() == 0)
		{			
			fprintf(stderr, "%6.2f %10.6f %10.0f\n", log(param_ptr->C)/log(2.0), cv_accuracy*100, cv_time);
		}		
		
		num_unchanged_w++;
		if(num_unchanged_w == 3)
		{
			*last_C = param_ptr->C;
		}	
		
	}	// for(param_ptr->C = start_C; param_ptr->C <= max_C; param_ptr->C *= ratio)
	
	free(target);
	for(int i=0; i<nr_fold; i++)
	{
		free(subdata[i].x);
		free(subdata[i].y);
	}
	free(subdata);
	for(int i = 0; i < nr_fold; i++)
		free(w_array[i]);
	free(w_array);
}

double test(model *model_ptr, data *data_ptr)
{
	int global_l = data_ptr->l;	
	int total_correct = 0;
	for(int i = 0; i < data_ptr->l; i++)
	{
		double label = predict(model_ptr, data_ptr->x[i]);
		if(label == data_ptr->y[i])
			total_correct++;
	}
	mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);	
	return (total_correct + 0.0) / global_l;
}

double predict(model *model_ptr, feature_node *x)
{
	int n = model_ptr->n;
	double wTx = 0;
	for( ; x->index != -1 && x->index <= n; x++)
		wTx += model_ptr->w[x->index - 1] * x->value;
	if(wTx > 0)
		return model_ptr->label[0];
	else
		return model_ptr->label[1];
}

model* malloc_model(int n)
{
	model *model_ptr = (model*)malloc(sizeof(model));
	model_ptr->n = n;
	model_ptr->w = (double*)malloc(sizeof(double) * n);
	for(int i = 0; i < n; i++)
		model_ptr->w[i] = 0.0;
	return model_ptr;
}

void free_model(model *model_ptr)
{
	free(model_ptr->w);
	free(model_ptr);
}

int save_model(const char *model_file_name, const model *model_ptr)
{	
	int n = model_ptr->n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp == NULL) 
		return -1;

	fprintf(fp, "label %d %d\n", model_ptr->label[0], model_ptr->label[1]);
	fprintf(fp, "n %d\n", n);

	fprintf(fp, "w\n");
	for(int i = 0; i < n; i++)
	{		
		fprintf(fp, "%.16g\n", model_ptr->w[i]);		
	}	

	if (ferror(fp) != 0 || fclose(fp) != 0) 
		return -1;
	else 
		return 0;
}

/*  static functions are only called in this file */
static double calc_start_C(const data *data, const parameter *param)
{
	int i;
	double xTx,max_xTx;
	max_xTx = 0;
	for(i=0; i<data->l; i++)
	{
		xTx = 0;
		feature_node *xi=data->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			xTx += val*val;
			xi++;
		}
		if(xTx > max_xTx)
			max_xTx = xTx;
	}
	int l = data->l;
	if(mpi_get_size() != 1)
	{
		mpi_allreduce(&max_xTx, 1, MPI_DOUBLE, MPI_MAX);	
		mpi_allreduce(&l, 1, MPI_INT, MPI_SUM);
	}
	
	double min_C = 1.0;

	min_C = 1.0 / (l * max_xTx);


	return pow( 2, floor(log(min_C) / log(2.0)) );
}

static void group_classes(const data *data_ptr, int *perm, int *label, int* positive)
{
	assert(data_ptr != 0 && label != 0 && positive != 0 && perm != 0);	
	
	int l = data_ptr->l;
	
	// step 1 : each node collects its own labels. 	
	set<int> label_set;
	for(int i = 0; i < l; i++)
		label_set.insert((int)data_ptr->y[i]);
	
	// step 2 : each node sends its labels to machine 0
	int buf[2];
	int size;	
	if(mpi_get_rank() == 0)
	{
		for(int rank = 1; rank < mpi_get_size(); rank++)
		{
			MPI_Status status;			
			MPI_Recv(&size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);			
			MPI_Recv(buf, size, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
			for(int j = 0; j < size; j++)
				label_set.insert(buf[j]);			
		}		
	}
	else
	{
		size = (int)label_set.size();
		assert(size <= 2);
		set<int>::iterator it;
		int i;
		for(it = label_set.begin(), i = 0; i < size; it++, i++)
			buf[i] = *it;
		MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(buf, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	
	// step 3 : machine 0 broadcast labels to each node
	int nr_class = (int)label_set.size();
	MPI_Bcast(&nr_class, 1, MPI_INT, 0, MPI_COMM_WORLD);
	assert(nr_class == 2);
	if(mpi_get_rank() == 0)
	{
		set<int>::iterator it;
		int i;
		for(it = label_set.begin(), i = 0; it != label_set.end(); it++, i++)
			label[i] = *it;
	}	
	MPI_Bcast(label, nr_class, MPI_INT, 0, MPI_COMM_WORLD);	

//	assert(label_set.size() == 2);
	if(label[0] == -1 && label[1] == 1)
		swap1(label[0], label[1]);	
	
	*positive = 0;
	for(int i = 0; i < l; i++)
	{
		if((int)data_ptr->y[i] == label[0])
			(*positive)++;
	}
	int i1 = 0, i2 = *positive;
	for(int i = 0; i < l; i++)
	{
		if((int)data_ptr->y[i] == label[0])
		{
			perm[i1] = i;
			i1++;
		}
		else
		{
			perm[i2] = i;
			i2++;
		}
	}
}

static void train_one(const data *data_ptr, const parameter *param_ptr, double *w)
{
	double primal_solver_tol;
	{
		int l = data_ptr->l;
		int pos = 0;
		int neg = 0;
		for(int i = 0; i < data_ptr->l; i++)
			if(data_ptr->y[i] > 0)
				pos++;
		mpi_allreduce(&pos, 1, MPI_INT, MPI_SUM);
		mpi_allreduce(&l, 1, MPI_INT, MPI_SUM);
		neg = l - pos;
		primal_solver_tol = param_ptr->eps;//(param_ptr->eps * max(min(pos,neg), 1)) / l;
	}
	
	classifier *classifier_ptr = new l2r_lr(data_ptr, param_ptr->C);
	TRON tron_obj(classifier_ptr, primal_solver_tol, param_ptr->eps_cg);	
	tron_obj.tron(w);
	delete classifier_ptr;
}

