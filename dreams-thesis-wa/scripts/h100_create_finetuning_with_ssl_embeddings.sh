#!/bin/bash
#SBATCH --job-name=dreams-finetune-ssl
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# Optional cluster/account override:
# SBATCH --account=<your_account>

set -euo pipefail

# ------------------------------------------------------------------
# Modules & conda (same pattern as fine-tune scripts)
# ------------------------------------------------------------------
module load 2024
module load Miniconda3/24.7.1-0

eval "$(conda shell.bash hook)"
conda activate dreams

# Export definitions so PRETRAINED and related paths are available.
$(python -c "from dreams.definitions import export; export()")

# ------------------------------------------------------------------
# Paths and run configuration
# ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$HOME/DreaMS}"
SCRIPT_PATH="$REPO_ROOT/dreams-thesis-wa/src/create_finetuning_with_ssl_embeddings.py"

# Default input/output under repository data directory.
INPUT_HDF5="${INPUT_HDF5:-$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5}"
OUTPUT_HDF5="${OUTPUT_HDF5:-$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning_with_ssl_embeddings.hdf5}"

BATCH_SIZE="${BATCH_SIZE:-512}"
N_HIGHEST_PEAKS="${N_HIGHEST_PEAKS:-100}"
OVERWRITE="${OVERWRITE:-1}"

echo ""
echo "=================================="
echo "DreaMS Finetuning SSL Embeddings"
echo "=================================="
echo "  Repo root            : $REPO_ROOT"
echo "  Script               : $SCRIPT_PATH"
echo "  Input HDF5           : $INPUT_HDF5"
echo "  Output HDF5          : $OUTPUT_HDF5"
echo "  Batch size           : $BATCH_SIZE"
echo "  n_highest_peaks      : $N_HIGHEST_PEAKS"
echo "=================================="
echo ""

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: script not found at $SCRIPT_PATH"
  exit 1
fi

if [ ! -f "$INPUT_HDF5" ]; then
  echo "Error: input file not found at $INPUT_HDF5"
  echo "Set an explicit path with INPUT_HDF5=/path/to/finetuning.hdf5"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_HDF5")"

# ------------------------------------------------------------------
# Run embedding generation
# ------------------------------------------------------------------
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

CMD=(
  python "$SCRIPT_PATH"
  --input-hdf5 "$INPUT_HDF5"
  --output-hdf5 "$OUTPUT_HDF5"
  --batch-size "$BATCH_SIZE"
  --n-highest-peaks "$N_HIGHEST_PEAKS"
  --device cuda
)

if [ "$OVERWRITE" = "1" ]; then
  CMD+=(--overwrite)
fi

echo "Running command: ${CMD[*]}"
srun --export=ALL --preserve-env "${CMD[@]}"

echo ""
echo "Done."
echo "Output written to: $OUTPUT_HDF5"

# Example usage:
# sbatch dreams-thesis-wa/scripts/h100_create_finetuning_with_ssl_embeddings.sh
# sbatch --export=INPUT_HDF5=$HOME/DreaMS/dreams-thesis-wa/data/processed/finetuning.hdf5,OUTPUT_HDF5=$HOME/DreaMS/dreams-thesis-wa/data/processed/finetuning_with_ssl_embeddings.hdf5,OVERWRITE=1 dreams-thesis-wa/scripts/h100_create_finetuning_with_ssl_embeddings.sh
