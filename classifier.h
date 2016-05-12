#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include "structures.h"

class classifier
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;
	virtual int get_nr_variables() = 0;
	virtual ~classifier(void){}
};

class l2r_lr : public classifier
{
public:
	l2r_lr(const data *data_ptr, double C);
	~l2r_lr();
	int get_nr_variables();
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double C;
	double *z;
	double *D;
	const data *data_ptr;
};


#endif

