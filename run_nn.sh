#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH -J "NN_Training_DL_FINAL"
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt

module load cuda/11.8.0/4w5kyjs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

~/miniconda3/bin/python3 combined_nn2.py

