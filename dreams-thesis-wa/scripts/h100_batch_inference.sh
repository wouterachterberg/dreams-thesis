#!/bin/bash
#SBATCH --job-name=dreams-axis2-infer
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
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
REPO_ROOT="${REPO_ROOT:-${DREAMS_REPO_ROOT:-$HOME/DreaMS}}"
SCRIPT_PATH="$REPO_ROOT/dreams-thesis-wa/scripts/h100_batch_inference.py"

# Persistent storage on cluster ($HOME)
PERSISTENT_CKPT_BASE_DIR="${CKPT_BASE_DIR:-}"
PERSISTENT_OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/dreams-thesis-wa/results/model_runs}"

# Optional explicit dataset overrides:
#   sbatch --export=PROBING_TEST_PATH=/path/probing_test.parquet,FINETUNING_HDF5_PATH=/path/finetuning.hdf5 ...
PROBING_TEST_PATH="${PROBING_TEST_PATH:-}"
FINETUNING_HDF5_PATH="${FINETUNING_HDF5_PATH:-}"

# Fast node-local/shared scratch
RUN_LABEL="${RUN_LABEL:-all6}"
SCRATCH_BASE="${SCRATCH_BASE:-${DREAMS_SCRATCH_BASE:-/scratch-shared/$USER}}"
SCRATCH_ROOT="$SCRATCH_BASE/dreams_axis2_infer_${RUN_LABEL}_${SLURM_JOB_ID}"
SCRATCH_DATA_DIR="$SCRATCH_ROOT/data"
SCRATCH_CKPT_DIR="$SCRATCH_ROOT/checkpoints"
SCRATCH_OUTPUT_ROOT="$SCRATCH_ROOT/model_runs"
COPY_CKPTS_TO_SCRATCH="${COPY_CKPTS_TO_SCRATCH:-0}"

mkdir -p "$SCRATCH_DATA_DIR" "$SCRATCH_CKPT_DIR" "$SCRATCH_OUTPUT_ROOT" "$PERSISTENT_OUTPUT_ROOT"

echo ""
echo "=================================="
echo "DreaMS Axis2 H100 Inference"
echo "=================================="
echo "  Repo root            : $REPO_ROOT"
echo "  Script               : $SCRIPT_PATH"
echo "  Persistent output    : $PERSISTENT_OUTPUT_ROOT"
echo "  Scratch root         : $SCRATCH_ROOT"
echo "=================================="
echo ""

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: script not found at $SCRIPT_PATH"
  exit 1
fi

# Auto-detect checkpoint root if not explicitly provided.
if [ -z "$PERSISTENT_CKPT_BASE_DIR" ]; then
  for candidate in \
    "$HOME/THESIS/model_checkpoints" \
    "$REPO_ROOT/dreams-thesis-wa/results/finetuning" \
    "$HOME/dreams-thesis-wa/results/finetuning"; do
    if [ -d "$candidate" ]; then
      PERSISTENT_CKPT_BASE_DIR="$candidate"
      break
    fi
  done
fi

if [ -z "$PERSISTENT_CKPT_BASE_DIR" ] || [ ! -d "$PERSISTENT_CKPT_BASE_DIR" ]; then
  echo "Error: checkpoint base dir not found."
  echo "Set it explicitly, e.g.:"
  echo "  sbatch --export=CKPT_BASE_DIR=/path/to/checkpoints dreams-thesis-wa/scripts/h100_batch_inference.sh"
  exit 1
fi

echo "Checkpoint source resolved to: $PERSISTENT_CKPT_BASE_DIR"

# Auto-detect dataset paths if not explicitly provided.
if [ -z "$PROBING_TEST_PATH" ]; then
  for candidate in \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/probing_test.parquet"; do
    if [ -f "$candidate" ]; then
      PROBING_TEST_PATH="$candidate"
      break
    fi
  done
fi

if [ -z "$FINETUNING_HDF5_PATH" ]; then
  for candidate in \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5"; do
    if [ -f "$candidate" ]; then
      FINETUNING_HDF5_PATH="$candidate"
      break
    fi
  done
fi

if [ -z "$PROBING_TEST_PATH" ] || [ ! -f "$PROBING_TEST_PATH" ]; then
  echo "Error: probing_test.parquet not found."
  echo "Looked in:"
  echo "  $REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet"
  echo "  $REPO_ROOT/dreams-thesis-wa/data/processed/probing_test.parquet"
  echo "Set explicit path with PROBING_TEST_PATH."
  exit 1
fi

if [ -z "$FINETUNING_HDF5_PATH" ] || [ ! -f "$FINETUNING_HDF5_PATH" ]; then
  echo "Error: finetuning.hdf5 not found."
  echo "Looked in:"
  echo "  $REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5"
  echo "  $REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5"
  echo "Set explicit path with FINETUNING_HDF5_PATH."
  exit 1
fi

echo "Probing dataset resolved to: $PROBING_TEST_PATH"
echo "Finetuning dataset resolved to: $FINETUNING_HDF5_PATH"

# ------------------------------------------------------------------
# Stage inputs to scratch
# ------------------------------------------------------------------
echo "Copying datasets to scratch..."
cp "$PROBING_TEST_PATH" "$SCRATCH_DATA_DIR/probing_test.parquet"
cp "$FINETUNING_HDF5_PATH" "$SCRATCH_DATA_DIR/finetuning.hdf5"

if [ "$COPY_CKPTS_TO_SCRATCH" = "1" ]; then
  echo "Copying checkpoints to scratch (recursive)..."
  rsync -a \
    --include='*/' \
    --include='*.ckpt' \
    --exclude='*' \
    "$PERSISTENT_CKPT_BASE_DIR/" "$SCRATCH_CKPT_DIR/"

  CKPT_COUNT=$(find "$SCRATCH_CKPT_DIR" -type f -name "*.ckpt" | wc -l | tr -d ' ')
  if [ "$CKPT_COUNT" -eq 0 ]; then
    echo "Error: no .ckpt files found under $PERSISTENT_CKPT_BASE_DIR"
    exit 1
  fi
  CKPT_SEARCH_DIR="$SCRATCH_CKPT_DIR"
  echo "Copied $CKPT_COUNT checkpoint files to scratch."
else
  CKPT_SEARCH_DIR="$PERSISTENT_CKPT_BASE_DIR"
  echo "Using checkpoint source in place (no scratch copy): $CKPT_SEARCH_DIR"
fi

# ------------------------------------------------------------------
# Run inference from scratch
# ------------------------------------------------------------------
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$SCRATCH_ROOT"

srun --export=ALL --preserve-env python "$SCRIPT_PATH" \
  --device cuda \
  --batch-size 1024 \
  --ckpt-base-dir "$CKPT_SEARCH_DIR" \
  --probing-test "$SCRATCH_DATA_DIR/probing_test.parquet" \
  --finetuning-hdf5 "$SCRATCH_DATA_DIR/finetuning.hdf5" \
  --output-root "$SCRATCH_OUTPUT_ROOT" \
  ${RUN_TAGS:+--run-tags "$RUN_TAGS"}

# ------------------------------------------------------------------
# Sync per-run results back to persistent storage
# ------------------------------------------------------------------
echo ""
echo "Syncing run outputs to persistent folders..."
for run_dir in "$SCRATCH_OUTPUT_ROOT"/*; do
  [ -d "$run_dir" ] || continue
  run_tag="$(basename "$run_dir")"
  mkdir -p "$PERSISTENT_OUTPUT_ROOT/$run_tag"
  rsync -a "$run_dir/" "$PERSISTENT_OUTPUT_ROOT/$run_tag/"
  echo "  Synced: $run_tag -> $PERSISTENT_OUTPUT_ROOT/$run_tag"
done

echo ""
echo "Cleaning up scratch: $SCRATCH_ROOT"
rm -rf "$SCRATCH_ROOT"
echo "Done."

# Optional: run only specific run tags
# sbatch --export=RUN_TAGS="morgan_2048_cos,morgan_2048_bce" dreams-thesis-wa/scripts/h100_batch_inference.sbatch
