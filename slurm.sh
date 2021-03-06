#!/bin/bash

#DO NOT TOUCH THIS SECTION BEGIN

####### CUSTOMIZE THIS SECTION FOR YOUR JOB
#SBATCH --job-name="braintumorresnet"
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:2
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --time=08:00:00

module load cuda/11.3.1
module load gcc/9.3.0
module load python/py38-anaconda-2021.05
source activate /panasas/scratch/grp-lsmatott/ve/condaclone
#DO NOT TOUCH THIS SECTION END

#Edit here BEGIN
python ./vgg19.py

#Edit here END

#DO NOT TOUCH THIS SECTION BEGIN
source deactivate /panasas/scratch/grp-lsmatott/ve/condaclone
module unload python/py38-anaconda-2021.05
module unload cuda/11.3.1
#DO NOT TOUCH THIS SECTION END