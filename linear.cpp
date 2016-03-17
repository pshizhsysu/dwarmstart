#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <mpi.h>
#include <set>
#include <map>
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w, int turn);
	void grad(double *w, double *g, int turn);
	void Hv(double *s, double *Hs, int turn);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}


double l2r_lr_fun::fun(double *w, int turn)
{
	int i;
	double f=0, reg=0;
	double *y=prob->y;
	int l=prob->l;
	int n=get_nr_variable();

	Xv(w, z);

	for(i=0;i<n;i++)
		reg += w[i]*w[i];
	reg /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}
	
	if(mpi_get_rank() == turn)
		f = 0;	
	mpi_allreduce(&f, 1, MPI_DOUBLE, MPI_SUM);
	
	f += reg;

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g, int turn)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int n=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);
	


	if(mpi_get_rank() == turn)
		for(int j = 0; j < n; j++)
			g[j] = 0;
			
	mpi_allreduce(g, n, MPI_DOUBLE, MPI_SUM);
	
	
	for(i=0;i<n;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs, int turn)
{
	int i;
	int l=prob->l;
	int n=get_nr_variable();
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);

	if(mpi_get_rank() == turn)
	{
		for(int j = 0; j < n; j++)
			Hs[j] = Hs[j];
	}
	
	
	mpi_allreduce(Hs, n, MPI_DOUBLE, MPI_SUM);	
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int n=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<n;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

double get_startC()
{
	return 1.0 / ( 1 << 30);	 // 2^-30
//	return 1;
}

static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int nr_class = 2;	

	std::map<int, int> label_map;
	int *label = Malloc(int, nr_class);
	label[0] = 1;
	label[1] = -1;
	label_map[1] = 0;
	label_map[-1] = 1;

	// The following codes are similar to the original LIBLINEAR
	int *start = Malloc(int, nr_class);
	int *count = Malloc(int, nr_class);
	for(int i=0;i<nr_class;i++)
		count[i] = 0;
	for(int i=0;i<l;i++)
		count[label_map[(int)prob->y[i]]]++;

	start[0] = 0;
	for(int i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(int i=0;i<l;i++)
	{
		perm[start[label_map[(int)prob->y[i]]]] = i;
		++start[label_map[(int)prob->y[i]]];
	}
	start[0] = 0;
	for(int i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn, int turn)
{
	int l = prob->l;
	double eps=param->eps;
	double primal_solver_tol;
	{		
		int global_pos = 0;
		int global_neg = 0;
		int global_l = l;
		for(int i=0; i<l; i++)
			if(prob->y[i] > 0)
				global_pos++;	

		if(mpi_get_rank() == turn)
			global_pos = global_l = 0;	
		
		mpi_allreduce(&global_pos, 1, MPI_INT, MPI_SUM);		
		mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);

		global_neg = global_l - global_pos;

		primal_solver_tol = (eps*max(min(global_pos,global_neg), 1))/global_l;
	}

	function *fun_obj=NULL;
	double *C = new double[l];
	for(int i = 0; i < l; i++)
	{
		if(prob->y[i] > 0)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
	fun_obj=new l2r_lr_fun(prob, C);
	TRON tron_obj(fun_obj, primal_solver_tol);
	tron_obj.set_print_string(liblinear_print_string);
	tron_obj.tron(w, turn);
	delete fun_obj;
	delete[] C;
}

/**
* w : initial model for training, can,t be NULL
* turn : data stored in node #turn are not for training, it's for testing
* 		 when turn is -1, it meams data in all nodes are for training
**/
model* train(const problem *prob_ptr, const parameter *param_ptr, const double* init_w, int turn)
{		
	int l = prob_ptr->l;
	int n = prob_ptr->n;	
	model *model_ptr = Malloc(model,1);

	if(prob_ptr->bias>=0)
		model_ptr->nr_feature=n-1;
	else
		model_ptr->nr_feature=n;
	model_ptr->param = *param_ptr;
	model_ptr->bias = prob_ptr->bias;

		int nr_class;
		int *label = NULL; // label[0] = 1, label[1] = -1
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob_ptr,&nr_class,&label,&start,&count,perm);

		model_ptr->nr_class=nr_class;
		model_ptr->label = Malloc(int,nr_class);
		for(int i=0;i<nr_class;i++)
			model_ptr->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(int i=0;i<nr_class;i++)
			weighted_C[i] = param_ptr->C;
		for(int i=0;i<param_ptr->nr_weight;i++)
		{
			int j;
			for(j=0;j<nr_class;j++)
				if(param_ptr->weight_label[i] == label[j])
					break;
			if(j == nr_class)
			{
				fprintf(stderr,"[rank %d] WARNING: class label %d specified in weight is not found\n",
						mpi_get_rank(), param_ptr->weight_label[i]);
			}
			else
				weighted_C[j] *= param_ptr->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(int i=0;i<l;i++)
			x[i] = prob_ptr->x[perm[i]];
		
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		int k;
		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];		
		//	if(nr_class == 2)
		//	{
				model_ptr->w=Malloc(double, n);	

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;				
			
				for(int i=0;i<n;i++)
						model_ptr->w[i] = init_w[i];
				train_one(&sub_prob, param_ptr, model_ptr->w, weighted_C[0], weighted_C[1], turn);
		//	}		

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	
	return model_ptr;
}

double cross_validation1(const struct problem* prob_ptr, const struct parameter* param_ptr)
{		
	int l = prob_ptr->l;
	int n = prob_ptr->n;		
	int nr_fold = mpi_get_size();
	model **model_ptr_array;
	model_ptr_array = Malloc(model*, nr_fold);
	double *init_w = Malloc(double, n);
	for(int j = 0; j < n; j++)
		init_w[j] = 0;
	
	for(int turn = 0; turn < mpi_get_size(); turn++)
		model_ptr_array[turn] = train(prob_ptr, param_ptr, init_w, turn);
				
	double *target = Malloc(double, l);
	int total_correct = 0;
	for(int i = 0; i < l; i++)
	{
		target[i] = predict(model_ptr_array[mpi_get_rank()], prob_ptr->x[i]);	
		if(target[i] == prob_ptr->y[i])
			++total_correct;
	}
	int global_l = l;	
	mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);	
	mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
	double cv_accuracy = (double)total_correct / global_l;				
	
	for(int i = 0; i < nr_fold; i++)
		free_and_destroy_model(&model_ptr_array[i]);
	free(model_ptr_array);
	free(target);
	free(init_w);
	
	return cv_accuracy;
}

double cross_validation2(const problem *prob_ptr, const parameter *param_ptr)
{	
	int nr_fold = mpi_get_size();
	int l = prob_ptr->l;
	int n = prob_ptr->n;
	double* target = Malloc(double, l);	
	double* init_w = Malloc(double, n);
	for(int i = 0; i < n; i++)
		init_w[i] = 0;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	
	// constructing sub-prob by training data : start
	struct problem *subprob = Malloc(struct problem, nr_fold);
	for(int i=0;i<nr_fold;i++)
	{
		subprob[i].bias = prob_ptr->bias;
		subprob[i].n = prob_ptr->n;
		subprob[i].l = l - (l / nr_fold + (l % nr_fold > i ? 1 : 0));
		subprob[i].x = Malloc(struct feature_node*, subprob[i].l);
		subprob[i].y = Malloc(double, subprob[i].l);
	}
	
	for(int i = 0; i < nr_fold; i++)
	{
		int ix = 0;
		for(int j = 0; j < l; j++)
		{
			if(j % nr_fold != i)
			{
				subprob[i].x[ix] = prob_ptr->x[j];
				subprob[i].y[ix] = prob_ptr->y[j];
				ix++;
			}
		}
	}
	// constructing sub-prob by training data : end
	
	// cross validation : start
	for(int i=0;i<nr_fold;i++)
	{
		struct model *submodel = train(&subprob[i],param_ptr,init_w,-1);	
		for(int k = 0; k < l - subprob[i].l; k++)
			target[i + k * nr_fold] = predict(submodel,prob_ptr->x[i + k * nr_fold]);
		free_and_destroy_model(&submodel);
	}
	int total_correct = 0;
	int global_l = l;
	for(int i = 0; i < l; i++)
		if(target[i] == prob_ptr->y[i])
			total_correct++;
	mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
	double cv_accuracy = (double)total_correct / global_l;
		
	free(target);
	free(init_w);
	for(int i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
	}
	free(subprob);
	return cv_accuracy;
}

void find_parameter_C1(const problem *prob_ptr, parameter *param_ptr, double start_C, double max_C, double *best_C, double *best_rate, FILE *output)
{	
	int l = prob_ptr->l;
	int n = prob_ptr->n;
	int nr_fold = mpi_get_size();
	double ratio = 2;
	int num_unchanged_w = -1;		
	double *target = Malloc(double, l);
	model **model_ptr_array;	
	model_ptr_array = Malloc(model*, nr_fold);
	double **w_array = Malloc(double*, nr_fold);
	for(int i = 0; i < nr_fold; i++)
	{	
		w_array[i] = Malloc(double, n);
		for(int j = 0; j < n; j++)
			w_array[i][j] = 0;
	}
	
	for(param_ptr->C = start_C; param_ptr->C <= max_C; param_ptr->C *= ratio)
	{	
		/* start : distributed cross validation */		
		double start = MPI_Wtime();
		for(int turn = 0; turn < nr_fold; turn++)
			model_ptr_array[turn] = train(prob_ptr, param_ptr, w_array[turn], turn);			
		
		for(int turn = 0; turn < nr_fold; turn++)
		{
			double norm_w_diff = 0;
			for(int j=0; j<n; j++)			
				norm_w_diff += (model_ptr_array[turn]->w[j] - w_array[turn][j])*(model_ptr_array[turn]->w[j] - w_array[turn][j]);				
			norm_w_diff = sqrt(norm_w_diff);
			if(norm_w_diff > 1e-15)
			{
				num_unchanged_w = -1;
				break;
			}
		}
		
		for(int turn = 0; turn < nr_fold; turn++)
		{			
			for(int i = 0; i < n; i++)
				w_array[turn][i] = model_ptr_array[turn]->w[i];
		}	
		
		int total_correct = 0;
		int global_l = l;		
		for(int i = 0; i < l; i++)
		{
			target[i] = predict(model_ptr_array[mpi_get_rank()], prob_ptr->x[i]);	
			if(target[i] == prob_ptr->y[i])
				++total_correct;
		}
		
		mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);		
		mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
	
		double cv_accuracy = (double)total_correct / global_l;
		/* End : distributed cross validation */	
		
		if(cv_accuracy > *best_rate)
		{
			*best_C = param_ptr->C;
			*best_rate = cv_accuracy;
		}						
		
		for(int i = 0; i < nr_fold; i++)
			free_and_destroy_model(&model_ptr_array[i]);
		
		double end = MPI_Wtime();
		double cv_time = (end - start);
		info("%6.2f%10.4f%8.2f\n", log(param_ptr->C)/log(2.0), cv_accuracy*100, cv_time);
		fprintf(output, "%6.2f%10.4f%8.2f\n", log(param_ptr->C)/log(2.0), cv_accuracy*100, cv_time);
	
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;	
	
	}	// for(param_ptr->C = start_C; param_ptr->C <= max_C; param_ptr->C *= ratio)
		
	for(int i = 0; i < nr_fold; i++)
		free(w_array[i]);
	free(w_array);
	free(model_ptr_array);
	free(target);
}

void find_parameter_C2(const problem *prob_ptr, parameter *param_ptr, double start_C, double max_C, double *best_C, double *best_rate, FILE *output)
{	
	int nr_fold = mpi_get_size();
	int l = prob_ptr->l;
	int n = prob_ptr->n;
	double* target = Malloc(double, l);		
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	
	// construting sub-prob by training data : start
	// master have {1,2,3,4,5,26,27,28,29,30}, then subprob[0].x has {2,3,4,5,27,28,29,30}, excluding {1,26}
	problem *subprob = Malloc(problem, nr_fold);
	for(int i=0;i<nr_fold;i++)
	{
		subprob[i].bias = prob_ptr->bias;
		subprob[i].n = prob_ptr->n;
		subprob[i].l = l - (l / nr_fold + (l % nr_fold > i ? 1 : 0));
		subprob[i].x = Malloc(struct feature_node*, subprob[i].l);
		subprob[i].y = Malloc(double, subprob[i].l);
	}
	
	for(int i = 0; i < nr_fold; i++)
	{
		int ix = 0;
		for(int j = 0; j < l; j++)
		{
			if(j % nr_fold != i)
			{
				subprob[i].x[ix] = prob_ptr->x[j];
				subprob[i].y[ix] = prob_ptr->y[j];
				ix++;
			}
		}
	}
	// construting sub-prob by training data : end
	
	double ratio = 2;
	int num_unchanged_w = -1;		
	double **w_array = Malloc(double*, nr_fold);
	for(int i = 0; i < nr_fold; i++)
	{	
		w_array[i] = Malloc(double, n);
		for(int j = 0; j < n; j++)
			w_array[i][j] = 0;
	}
	double* init_w = Malloc(double, n);
	for(int i = 0; i < n; i++)
		init_w[i] = 0;
	
	for(param_ptr->C = start_C; param_ptr->C <= max_C; param_ptr->C *= ratio)
	{	
		double start = MPI_Wtime();
		for(int i=0;i<nr_fold;i++)
		{
			struct model *submodel = train(&subprob[i],param_ptr,w_array[i],-1);	
			for(int k = 0; k < l - subprob[i].l; k++)
				target[i + k * nr_fold] = predict(submodel,prob_ptr->x[i + k * nr_fold]);
			
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
			
			for(int k = 0; k < n; k++)
				w_array[i][k] = submodel->w[k];	
			free_and_destroy_model(&submodel);
		}	// for(int i=0;i<nr_fold;i++)
		
		int total_correct = 0;
		int global_l = l;
		for(int i = 0; i < l; i++)
			if(target[i] == prob_ptr->y[i])
				total_correct++;
		mpi_allreduce(&total_correct, 1, MPI_INT, MPI_SUM);
		mpi_allreduce(&global_l, 1, MPI_INT, MPI_SUM);
		double cv_accuracy = (double)total_correct / global_l;		
		if(cv_accuracy > *best_rate)
		{
			*best_rate = cv_accuracy;
			*best_C = param_ptr -> C;
		}
		double end = MPI_Wtime();
		double cv_time = (end - start);
		info("%6.2f%10.4f%8.2f\n", log(param_ptr->C)/log(2.0), cv_accuracy*100, cv_time);
		fprintf(output, "%6.2f%10.4f%8.2f\n", log(param_ptr->C)/log(2.0), cv_accuracy*100, cv_time);
		
		
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;
		
	}	// for(param_ptr->C = start_C; param_ptr->C <= max_C; param_ptr->C *= ratio)
	
	free(target);
	for(int i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
	}
	free(subprob);
	for(int i = 0; i < nr_fold; i++)
		free(w_array[i]);
	free(w_array);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(check_regression_model(model_))
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	int flag;
	while(1)
	{
		flag = fscanf(fp,"%80s",cmd);
		if(flag == EOF)
			printf("load model EOF error\n");
		if(strcmp(cmd,"solver_type")==0)
		{
			flag = fscanf(fp,"%80s",cmd);
			if(flag == EOF)
				printf("load model EOF error\n");
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"[rank %d] unknown solver type.\n", mpi_get_rank());

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			flag = fscanf(fp,"%d",&nr_class);
			if(flag == EOF)
				printf("load model EOF error\n");
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			flag = fscanf(fp,"%d",&nr_feature);
			if(flag == EOF)
				printf("load model EOF error\n");
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			flag = fscanf(fp,"%lf",&bias);
			if(flag == EOF)
				printf("load model EOF error\n");
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
			{
				flag = fscanf(fp,"%d",&model_->label[i]);
				if(flag == EOF)
					printf("load model EOF error\n");
			}
		}
		else
		{
			fprintf(stderr,"[rank %d] unknown text in model file: [%s]\n",mpi_get_rank(),cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, n*nr_w);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
		{
			flag = fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
			if(flag == EOF)
				printf("load model EOF error\n");
		}
		flag = fscanf(fp, "\n");
		if(flag == EOF)
			printf("load model EOF error\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	int solver_type = model_->param.solver_type;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(check_regression_model(model_))
		return w[idx];
	else
	{
		if(label_idx < 0 || label_idx >= nr_class)
			return 0;
		if(nr_class == 2 && solver_type != MCSVM_CS)
		{
			if(label_idx == 0)
				return w[idx];
			else
				return -w[idx];
		}
		else
			return w[idx*nr_class+label_idx];
	}
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
	if(feat_idx > model_->nr_feature)
		return 0;
	return get_w_value(model_, feat_idx-1, label_idx);
}

double get_decfun_bias(const struct model *model_, int label_idx)
{
	int bias_idx = model_->nr_feature;
	double bias = model_->bias;
	if(bias <= 0)
		return 0;
	else
		return bias*get_w_value(model_, bias_idx, label_idx);
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
	  && param->solver_type != L2R_L2LOSS_SVC
	  && param->solver_type != L2R_L2LOSS_SVC_DUAL
	  && param->solver_type != L2R_L1LOSS_SVC_DUAL)
		return "unknown solver type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_L2LOSS_SVR ||
			model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
			model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
