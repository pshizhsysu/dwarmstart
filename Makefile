CXX = mpic++
CC = mpicc

all : train

train : train.c linear.o tron.o classifier.o blas/blas.a
	$(CXX) -o train train.c linear.o tron.o classifier.o blas/blas.a
	
linear.o : linear.cpp linear.h tron.h classifier.h structures.h mpi_fun.h
	$(CXX) -c -o linear.o linear.cpp
	
tron.o : tron.cpp tron.h classifier.h structures.h mpi_fun.h
	$(CXX) -c -o tron.o tron.cpp
	
classifier.o : classifier.cpp classifier.h structures.h mpi_fun.h
	$(CXX) -c -o classifier.o classifier.cpp
	
clean :
	rm linear.o tron.o classifier.o