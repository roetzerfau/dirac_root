#!/bin/bash -l

#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --time=1-00:00:00
#SBATCH --job-name=test13_12
#SBATCH --export=NONE
#SBATCH --output=test13_12.txt
#SBATCH --mem=50G



module load user-spack
module load dealii
module load openmpi
make
srun LDGPoissonCoupled



