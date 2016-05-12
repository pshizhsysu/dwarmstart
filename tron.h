#ifndef _TRON_H
#define _TRON_H

#include "classifier.h"

class TRON
{
public:
	TRON(classifier *classifier_ptr, double eps, double eps_cg);
	~TRON();
	void tron(double *w);
private:	
	double eps;
	double eps_cg;
	classifier *classifier_ptr;		
	int trcg(double delta, double *g, double *s, double *r);
};

#endif
