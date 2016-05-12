#ifndef _MPI_FUN_H
#define _MPI_FUN_H

#include <mpi.h>
#include <vector>

static int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

static int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

static void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}

template<typename T>
static void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
	std::vector<T> buf_reduced(count);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
}

template <class T> 
static inline void swap1(T& x, T& y) 
{ 
	T t = x;
	x = y; 
	y = t; 
}

#endif

