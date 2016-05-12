#!/bin/bash

mpiexec -n 1 --machinefile machinefile ./train /home/jing/data/ijcnn 

/home/jing/cluster/liblinear-2.1/train -s 0 /home/jing/data/ijcnn









