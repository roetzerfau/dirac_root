#!/bin/bash –l
make
nohup mpirun -np 4 LDGPoissonCoupled > output.log 2>&1
#nohup make run > output.log 2>&1
