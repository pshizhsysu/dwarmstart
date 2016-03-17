#ifndef _TRON_H
#define _TRON_H

class function
{
public:
	virtual double fun(double *w, int turn) = 0 ;
	virtual void grad(double *w, double *g, int turn) = 0 ;
	virtual void Hv(double *s, double *Hs, int turn) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();

	void tron(double *w, int turn);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trcg(double delta, double *g, double *s, double *r, int turn);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
