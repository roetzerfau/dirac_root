#!/bin/bash –l
make
nohup mpirun -np 2 LDGPoissonCoupled > output.log 2>&1