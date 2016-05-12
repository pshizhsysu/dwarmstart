#ifndef _STRUCTURES_H
#define _STRUCTURES_H

struct feature_node
{
	int index;
	double value;
};

struct data
{
	int l;
	int n;
	double *y;
	struct feature_node **x; 
};

struct parameter
{	
	double eps;	        
	double eps_cg;
	double C;
};

struct model
{	
	double *w;
	int label[2];	
	int n;
};

#endif

