#!/bin/bash

####### CUSTOMIZE THIS SECTION FOR YOUR JOB
#SBATCH --job-name="pikapika"
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_v100-pcie-16gb:1
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --time=08:00:00

echo $CUDA_VISIBLE_DEVICES

module load cuda/11.0
module load python/py38-anaconda-2021.05
source activate /panasas/scratch/grp-lsmatott/ve/condaclone

python ./vgg19.py

source deactivate /panasas/scratch/grp-lsmatott/ve/condaclone
module unload python/py38-anaconda-2021.05
module unload cuda/11.0