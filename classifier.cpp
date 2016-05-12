#include <stdlib.h>
#include <math.h>
#include "classifier.h"
#include "structures.h"
#include "mpi_fun.h"

using namespace std;

l2r_lr::l2r_lr(const data *data_ptr, double C)
{
	this->data_ptr = data_ptr;
	z = (double*)malloc(sizeof(double) * data_ptr->l);
	D = (double*)malloc(sizeof(double) * data_ptr->l);
	this->C = C;
}

l2r_lr::~l2r_lr()
{
	free(z);
	free(D);
}

int l2r_lr::get_nr_variables()
{
	return data_ptr->n;
}

double l2r_lr::fun(double *w)
{
	double f = 0, reg = 0;
	double *y = data_ptr->y;
	int l = data_ptr->l;
	int n = data_ptr->n;

	Xv(w, z);

	for(int i = 0; i < n; i++)
		reg += w[i] * w[i];
	reg /= 2.0;
	for(int i = 0; i < l; i++)
	{
		double yz = y[i] * z[i];
		if (yz >= 0)
			f += C * log(1 + exp(-yz));
		else
			f += C * (-yz + log(1 + exp(yz)));
	}	

	if(mpi_get_size() != 1)
		mpi_allreduce(&f, 1, MPI_DOUBLE, MPI_SUM);
	
	f += reg;

	return(f);
}

void l2r_lr::grad(double *w, double *g)
{
	double *y = data_ptr->y;
	int l = data_ptr->l;
	int n = data_ptr->n;

	for(int i = 0; i < l; i++)
	{
		z[i] = 1 / (1 + exp(-y[i] * z[i]));
		D[i] = z[i] * (1 - z[i]);
		z[i] = C * (z[i] - 1) * y[i];
	}
	XTv(z, g);
	if(mpi_get_size() != 1)		
		mpi_allreduce(g, n, MPI_DOUBLE, MPI_SUM);
	
	for(int i = 0; i < n; i++)
		g[i] = w[i] + g[i];
}

void l2r_lr::Hv(double *s, double *Hs)
{	
	int l = data_ptr->l;
	int n = data_ptr->n;
	double *wa = (double*)malloc(sizeof(double) * l);

	Xv(s, wa);
	for(int i = 0; i < l; i++)
		wa[i] = C * D[i] * wa[i];

	XTv(wa, Hs);
	
	if(mpi_get_size() != 1)
		mpi_allreduce(Hs, n, MPI_DOUBLE, MPI_SUM);	
	for(int i = 0; i < n; i++)
		Hs[i] = s[i] + Hs[i];
	
	free(wa);
}

void l2r_lr::Xv(double *v, double *Xv)
{
	int l = data_ptr->l;
	feature_node **x = data_ptr->x;

	for(int i = 0; i < l; i++)
	{
		feature_node *s = x[i];
		Xv[i] = 0;
		while(s->index != -1)
		{
			Xv[i] += v[s->index-1] * s->value;
			s++;
		}
	}
}

void l2r_lr::XTv(double *v, double *XTv)
{
	int l = data_ptr->l;
	int n = data_ptr->n;
	feature_node **x = data_ptr->x;

	for(int i = 0; i < n; i++)
		XTv[i] = 0;
	for(int i = 0; i < l; i++)
	{
		feature_node *s = x[i];
		while(s->index != -1)
		{
			XTv[s->index - 1] += v[i] * s->value;
			s++;
		}
	}
}
