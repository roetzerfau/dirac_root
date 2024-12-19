#!/bin/bash –l
make
nohup mpirun -np 2 LDGPoissonCoupled > output_test.log 2>&1
#nohup make run > output_3D1Dconst_graded_mesh.log 2>&1
