#!/bin/bash –l
make
nohup mpirun -np 8 LDGPoissonCoupled > output_3D1Dconst.log 2>&1
#nohup make run > output_2D.log 2>&1
