#!/bin/bash –l
make
#nohup mpirun -np 4 LDGPoissonCoupled > output_3D1Dconst_graded_mesh.log 2>&1
nohup make run > output_3D1Dconst_graded_mesh.log 2>&1
