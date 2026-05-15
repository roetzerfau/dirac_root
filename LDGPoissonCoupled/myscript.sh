#!/bin/bash –l
#make
#nohup /usr/bin/time -v mpirun -np 4 LDGPoissonCoupled > output_30_04_3D.log 2>&1
nohup make run > output_3D1D_1DGap.log 2>&1
