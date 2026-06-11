#!/bin/bash
#SBATCH --job-name=DreaMS_ft_maccs_bce
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# MACCS 166 + BCE loss
export FP_OBJECTIVE=fp_maccs_166
export FP_LOSS=bce_logits
# export FP_POS_WEIGHT=6  # Uncomment for pos_weight (MACCS ~15% density)

exec bash "$HOME/DreaMS/dreams-thesis-wa/scripts/fine_tune_test.sh"
