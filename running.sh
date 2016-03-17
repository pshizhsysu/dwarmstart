#!/bin/bash

# split data data_file into 5 sub-data, sub-data named as data_file.sub, stored in /home/jing/dis_data/
	#./split.py machinefile /home/jing/data/heart_scale
	#./split.py machinefile /home/jing/data/madelon 
	#./split.py machinefile /home/jing/data/ijcnn 
	#./split.py machinefile /home/jing/data/webspam 
	#./split.py machinefile /home/jing/data/rcv1 
	#./split.py machinefile /home/jing/data/yahoo-japan 
	#./split.py machinefile /home/jing/data/news20 

# distributed cross validation
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/heart_scale.sub
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/madelon.sub
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/ijcnn.sub
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/webspam.sub
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/rcv1.sub
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/yahoo-japan.sub
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -v 2 /home/jing/dis_data/news20.sub

# distributed warm start 1
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/heart_scale.sub
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/madelon.sub.1
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/ijcnn.sub.1
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/webspam.sub.1
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/rcv1.sub.1
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/yahoo-japan.sub
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 1 /home/jing/dis_data/news20.sub.1
	
# distributed warm start 2
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/heart_scale.sub
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/madelon.sub.2
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/ijcnn.sub.2
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/webspam.sub.2
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/rcv1.sub.2
	#mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/yahoo-japan.sub
	mpiexec -n 5 --machinefile machinefile ./train -s 0 -C 2 /home/jing/dis_data/news20.sub.2

# do training 
	#mpiexec -n 5 --machinefile machinefile ./train /home/jing/dis_data/heart_scale.sub /home/jing/models/heart_scale.dis.model
	#mpiexec -n 5 --machinefile machinefile ./train /home/jing/dis_data/madelon.sub /home/jing/models/madelon.dis.model
	#mpiexec -n 5 --machinefile machinefile ./train /home/jing/dis_data/ijcnn.sub /home/jing/models/ijcnn.dis.model
	#mpiexec -n 5 --machinefile machinefile ./train /home/jing/dis_data/webspam.sub /home/jing/models/webspam.dis.model
	#mpiexec -n 5 --machinefile machinefile ./train /home/jing/dis_data/rcv1.sub /home/jing/models/rcv1.dis.model
	#mpiexec -n 5 --machinefile machinefile ./train /home/jing/dis_data/news20.sub /home/jing/models/news20.dis.model

# doing prediction
	#mpiexec -n 5 --machinefile machinefile ./predict /home/jing/dis_data/heart_scale.sub /home/jing/dis_data/heart_scale.sub.model /home/jing/dis_data/heart_scale.sub.out 
