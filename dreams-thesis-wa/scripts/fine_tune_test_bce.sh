#!/bin/bash
#SBATCH --job-name=DreaMS_ft_morgan_bce
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# Morgan 2048 + BCE loss
export FP_OBJECTIVE=fp_morgan_2048
export FP_LOSS=bce_logits
# export FP_POS_WEIGHT=44  # Uncomment for pos_weight (e.g. ~2.2% positives)

exec bash "$HOME/DreaMS/dreams-thesis-wa/scripts/fine_tune_test.sh"
