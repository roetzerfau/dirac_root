#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=testjob
#SBATCH --export=NONE
#SBATCH --output=output_22_05.txt
#SBATCH --mem=230G
./my_container.sif 
