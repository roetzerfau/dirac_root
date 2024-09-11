#!/bin/bash –l
make
nohup mpirun -np 40 LDGPoissonCoupled > output.log 2>&1
