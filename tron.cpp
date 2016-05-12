#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "tron.h"
#include "classifier.h"
#include "mpi_fun.h"


using namespace std;

extern "C"
{
	extern double dnrm2_(int *, double *, int *);
	extern double ddot_(int *, double *, int *, double *, int *);
	extern int daxpy_(int *, double *, double *, int *, double *, int *);
	extern int dscal_(int *, double *, double *, int *);
}

TRON::TRON(classifier *classifier_ptr, double eps, double eps_cg)
{
	this->classifier_ptr = classifier_ptr;
	this->eps = eps;
	this->eps_cg = eps_cg;	
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	// Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = classifier_ptr->get_nr_variables();
	int i, cg_iter;
	double gnorm, gnorm0, delta, snorm, one=1.0;
	double alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	double *s = (double*)malloc(sizeof(double) * n);
	double *r = (double*)malloc(sizeof(double) * n);	
	double *g = (double*)malloc(sizeof(double) * n);

	// calculate gradient norm at w=0 for stopping condition.
	double *w_new = (double*)malloc(sizeof(double) * n);
	for (i = 0; i < n; i++)
		w_new[i] = 0;
	classifier_ptr->fun(w_new);
	classifier_ptr->grad(w_new, g);
	gnorm0 = dnrm2_(&n, g, &inc);	
	
	f = classifier_ptr->fun(w);
	classifier_ptr->grad(w, g);
	gnorm = dnrm2_(&n, g, &inc);
	delta = gnorm;
	
	if(gnorm <= eps * gnorm0)
		search = 0;
	
	iter = 1;
	while (iter <= 100 && search)
	{		
		cg_iter = trcg(delta, g, s, r);
		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);
		gs = ddot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
		fnew = classifier_ptr->fun(w_new);		
		actred = f - fnew;			
		
		// On the first iteration, adjust the initial step bound.
		snorm = dnrm2_(&n, s, &inc);
		if (iter == 1)
			delta = min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
		else
			delta = max(delta, min(alpha*snorm, sigma3*delta));

		if(mpi_get_rank() == 0)
			fprintf(stderr, "iter %2d eps %g eps_cg %g act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e |g0| %5.3e CG %3d\n", 
							iter, eps, eps_cg, actred, prered, delta, f, gnorm, gnorm0, cg_iter);

		if (actred > eta0 * prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(double) * n);	
			f = fnew;
			classifier_ptr->grad(w, g);

			gnorm = dnrm2_(&n, g, &inc);
			if (gnorm <= eps * gnorm0)
				break;
		}
		
		if (f < -1.0e+32)
		{
			if(mpi_get_rank() == 0)
				fprintf(stderr, "WARNING: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 0 && prered <= 0)
		{
			if(mpi_get_rank() == 0)
				fprintf(stderr, "WARNING: actred and prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) && fabs(prered) <= 1.0e-12*fabs(f))
		{
			if(mpi_get_rank() == 0)
				fprintf(stderr, "WARNING: actred and prered too small\n");
			break;
		}
	}
	if(mpi_get_rank() == 0)
		fprintf(stderr, "Outer iteration : %d, f = %5.3e, |g| = %5.3e\n", iter - 1, f, gnorm);
	free(g);
	free(r);
	free(w_new);
	free(s);
}

int TRON::trcg(double delta, double *g, double *s, double *r)
{	
	int i, inc = 1;
	int n = classifier_ptr->get_nr_variables();
	double one = 1;
	double *d = (double*)malloc(sizeof(double) * n);
	double *Hd = (double*)malloc(sizeof(double) * n);
	double rTr, rnewTrnew, alpha, beta, cgtol;	
	
	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}	
	cgtol = eps_cg * dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
//	while(cg_iter < 10)	
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		classifier_ptr->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		if (dnrm2_(&n, s, &inc) > delta)
		{
		//	if(mpi_get_rank() == 0)
		//		fprintf(stderr, "cg reaches trust region boundary\n");
			alpha = -alpha;
			daxpy_(&n, &alpha, d, &inc, s, &inc);

			double std = ddot_(&n, s, &inc, d, &inc);
			double sts = ddot_(&n, s, &inc, s, &inc);
			double dtd = ddot_(&n, d, &inc, d, &inc);
			double dsq = delta*delta;
			double rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			daxpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			daxpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	free(d);
	free(Hd);

	return(cg_iter);
}

