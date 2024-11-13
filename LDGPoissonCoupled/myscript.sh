#!/bin/bash –l
make
nohup mpirun -np 2 LDGPoissonCoupled > output.log 2>&1
#nohup make run > output_2D.log 2>&1
