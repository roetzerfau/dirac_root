#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=testjob
#SBATCH --export=NONE
#SBATCH --output=output.txt

./my_container.sif 