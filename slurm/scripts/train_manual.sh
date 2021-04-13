#!/bin/bash

#SBATCH -o ../outputs/manual_%A.%a.out
#SBATCH -e ../outputs/manual_%A.%a.err
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -w node009
#SBATCH --mail-type=all
#SBATCH --mail-user=ssebastian94@gwu.edu
#SBATCH --array=0-4

module load anaconda/2020.07
source /modules/apps/anaconda3/etc/profile.d/conda.sh
conda activate malaria_classification
/home/ssebastian94/.conda/envs/malaria_classification/bin/python /home/ssebastian94/malaria_classification/src/model_trainers/0_manual.py