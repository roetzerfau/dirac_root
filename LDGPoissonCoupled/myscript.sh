#!/bin/bash –l
make
nohup mpirun -np 16 valgrind LDGPoissonCoupled > output_3D1Dconst_15_11.log 2> valgrind.%p.log
#nohup make run > output_2D.log 2>&1
