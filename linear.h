#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include <mpi.h>
#include "structures.h"

void train(const data *data_ptr, const parameter *param_ptr, model* model_ptr);
double cross_validation(const data *data_ptr, const parameter *param_ptr, int nr_fold);
void find_parameter_C(const data *data_ptr, parameter *param_ptr, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate, double *last_C);			
double test(model *model_ptr, data *data_ptr);
double predict(model *model_ptr, feature_node *x);
model* malloc_model(int n);
void free_model(model *model_ptr);
int save_model(const char *model_file_name, const model *model_ptr);

#endif

