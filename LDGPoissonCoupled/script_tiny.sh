#!/bin/bash -l
 	
#SBATCH --ntasks=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=testjob
#SBATCH --export=NONE
#SBATCH --output=outputtiny.txt



module load user-spack
module load dealii
module load openmpi
make
srun LDGPoissonCoupled

