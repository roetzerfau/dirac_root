#!/bin/bash –l
make
nohup mpirun -np 10 LDGPoissonCoupled > output_3D1Dconst_uncoupled_ram_works.log 2>&1
#nohup make run > output_2D.log 2>&1
