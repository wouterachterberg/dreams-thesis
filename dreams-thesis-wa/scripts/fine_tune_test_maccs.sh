#!/bin/bash
#SBATCH --job-name=DreaMS_ft_maccs_cos
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# MACCS 166 + cosine loss
export FP_OBJECTIVE=fp_maccs_166
export FP_LOSS=cos

exec bash "$HOME/DreaMS/dreams-thesis-wa/scripts/fine_tune_test.sh"
