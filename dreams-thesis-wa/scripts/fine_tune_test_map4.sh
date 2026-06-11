#!/bin/bash
#SBATCH --job-name=DreaMS_ft_map4_cos
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# MAP4 2048 + cosine loss
# NOTE: Requires `map4` package on Snellius (pip install map4).
#       MAP4 fingerprints are computed on-the-fly from SMILES during training.
#       If map4 is not available, precompute with:
#         python dreams-thesis-wa/src/add_map4_fingerprints.py data/processed/finetuning.hdf5
export FP_OBJECTIVE=fp_map4_2048
export FP_LOSS=cos

exec bash "$HOME/DreaMS/dreams-thesis-wa/scripts/fine_tune_test.sh"
