#!/bin/bash
#SBATCH --job-name=DreaMS_ft_map4_bce
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# MAP4 2048 + BCE loss
# NOTE: Requires `map4` package on Snellius (pip install map4).
#       See fine_tune_test_map4.sh for precompute instructions.
export FP_OBJECTIVE=fp_map4_2048
export FP_LOSS=bce_logits
# export FP_POS_WEIGHT=44  # Uncomment for pos_weight

exec bash "$HOME/DreaMS/dreams-thesis-wa/scripts/fine_tune_test.sh"
