#!/bin/bash –l
make
nohup mpirun -np 12 LDGPoissonCoupled > output_3D1Dconst_uncoupled.log 2>&1
#nohup make run > output_2D.log 2>&1
