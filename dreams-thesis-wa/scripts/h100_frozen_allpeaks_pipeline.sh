#!/bin/bash
#SBATCH --job-name=dreams-frozen-allpeaks
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out

# Optional cluster/account override:
# SBATCH --account=<your_account>

set -euo pipefail

# ------------------------------------------------------------------
# Modules & conda (same pattern as other DreaMS jobs)
# ------------------------------------------------------------------
module load 2024
module load Miniconda3/24.7.1-0

eval "$(conda shell.bash hook)"
conda activate dreams

# Export definitions so PRETRAINED and related paths are available.
$(python -c "from dreams.definitions import export; export()")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-${DREAMS_REPO_ROOT:-$HOME/DreaMS}}"
SRC_DIR="$REPO_ROOT/dreams-thesis-wa/src"
SCRIPTS_DIR="$REPO_ROOT/dreams-thesis-wa/scripts"

# Persistent paths (home/project storage)
PERSIST_DATA_DIR="${PERSIST_DATA_DIR:-$REPO_ROOT/dreams-thesis-wa/data/processed}"
PERSIST_RESULTS_MODEL_RUNS="${PERSIST_RESULTS_MODEL_RUNS:-$REPO_ROOT/dreams-thesis-wa/results/model_runs}"
PERSIST_RESULTS_FROZEN_ALLPEAKS="${PERSIST_RESULTS_FROZEN_ALLPEAKS:-$REPO_ROOT/dreams-thesis-wa/results/frozen_allpeaks_baselines}"

# Inputs
resolve_first_existing_file () {
  for candidate in "$@"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

FINETUNING_CANDIDATES="$PERSIST_DATA_DIR/finetuning.hdf5 | $REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5 | $REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5"
WITH_SSL_CANDIDATES="$PERSIST_DATA_DIR/finetuning_with_ssl_embeddings.hdf5 | $REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5 | $REPO_ROOT/dreams-thesis-wa/data/processed/finetuning_with_ssl_embeddings.hdf5"
PROBING_CANDIDATES="$PERSIST_DATA_DIR/probing_test.parquet | $REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet | $REPO_ROOT/dreams-thesis-wa/data/processed/probing_test.parquet"
FP_CACHE_CANDIDATES="$PERSIST_DATA_DIR/fingerprint_cache.npz | $REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/fingerprint_cache.npz | $REPO_ROOT/dreams-thesis-wa/data/processed/fingerprint_cache.npz"

if [ -z "${INPUT_FINETUNING_HDF5:-}" ]; then
  INPUT_FINETUNING_HDF5="$(resolve_first_existing_file \
    "$PERSIST_DATA_DIR/finetuning.hdf5" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5" || true)"
fi

if [ -z "${INPUT_FINETUNING_WITH_SSL_HDF5:-}" ]; then
  INPUT_FINETUNING_WITH_SSL_HDF5="$(resolve_first_existing_file \
    "$PERSIST_DATA_DIR/finetuning_with_ssl_embeddings.hdf5" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning_with_ssl_embeddings.hdf5" || true)"
fi

if [ -z "${INPUT_PROBING_TEST:-}" ]; then
  INPUT_PROBING_TEST="$(resolve_first_existing_file \
    "$PERSIST_DATA_DIR/probing_test.parquet" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/probing_test.parquet" || true)"
fi

if [ -z "${INPUT_FP_CACHE:-}" ]; then
  INPUT_FP_CACHE="$(resolve_first_existing_file \
    "$PERSIST_DATA_DIR/fingerprint_cache.npz" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/fingerprint_cache.npz" \
    "$REPO_ROOT/dreams-thesis-wa/data/processed/fingerprint_cache.npz" || true)"
fi

validate_input_file () {
  local label="$1"
  local path="$2"
  local candidates="$3"

  if [ -z "$path" ]; then
    echo "Error: missing required input for $label."
    echo "  Tried: $candidates"
    return 1
  fi

  if [ ! -f "$path" ]; then
    echo "Error: missing required input for $label: $path"
    echo "  Tried: $candidates"
    return 1
  fi

  return 0
}

# Runtime knobs
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-256}"
# All-peaks tensors are larger than precursor-only, but H100 jobs should handle
# substantially larger batches than 64. Default to 256 for better throughput.
# Override at submit time if needed, e.g. --export=HEAD_BATCH_SIZE=512.
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-256}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
MAX_EPOCHS="${MAX_EPOCHS:-103}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
EMB_DTYPE="${EMB_DTYPE:-float16}"     # float16 recommended
EMB_FOLDS="${EMB_FOLDS:-all}"          # all or train,val
OVERWRITE_PEAKS="${OVERWRITE_PEAKS:-1}"
USE_HDF5_PEAK_MASK="${USE_HDF5_PEAK_MASK:-1}"

# Optional run subset (comma-separated tags from defaults in inference script)
RUN_TAGS="${RUN_TAGS:-}"

# Scratch
PIPELINE_LABEL="${PIPELINE_LABEL:-allpeaks_${SLURM_JOB_ID}}"
SCRATCH_BASE="${SCRATCH_BASE:-${DREAMS_SCRATCH_BASE:-/scratch-shared/$USER}}"
SCRATCH_ROOT="$SCRATCH_BASE/dreams_frozen_allpeaks_${PIPELINE_LABEL}"
SCRATCH_DATA_DIR="$SCRATCH_ROOT/data"
mkdir -p "$SCRATCH_DATA_DIR" "$PERSIST_RESULTS_MODEL_RUNS" "$PERSIST_RESULTS_FROZEN_ALLPEAKS"

echo ""
echo "=================================="
echo "DreaMS Frozen-AllPeaks Pipeline"
echo "=================================="
echo "Repo root                  : $REPO_ROOT"
echo "Scratch root               : $SCRATCH_ROOT"
echo "Finetuning input           : $INPUT_FINETUNING_HDF5"
echo "With-SSL input             : $INPUT_FINETUNING_WITH_SSL_HDF5"
echo "Probing test input         : $INPUT_PROBING_TEST"
echo "Fingerprint cache          : $INPUT_FP_CACHE"
echo "Persistent model_runs out  : $PERSIST_RESULTS_MODEL_RUNS"
echo "Persistent allpeaks out    : $PERSIST_RESULTS_FROZEN_ALLPEAKS"
echo "=================================="
echo ""

# ------------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------------
validate_input_file "finetuning.hdf5" "$INPUT_FINETUNING_HDF5" "$FINETUNING_CANDIDATES" || exit 1
validate_input_file "finetuning_with_ssl_embeddings.hdf5" "$INPUT_FINETUNING_WITH_SSL_HDF5" "$WITH_SSL_CANDIDATES" || exit 1
validate_input_file "probing_test.parquet" "$INPUT_PROBING_TEST" "$PROBING_CANDIDATES" || exit 1

if [ -n "${INPUT_FP_CACHE:-}" ] && [ ! -f "$INPUT_FP_CACHE" ]; then
  echo "Warning: provided fingerprint cache path does not exist: $INPUT_FP_CACHE"
  echo "Will auto-generate fingerprint_cache.npz from finetuning.hdf5"
  INPUT_FP_CACHE=""
fi

if [ -z "${INPUT_FP_CACHE:-}" ]; then
  echo "Fingerprint cache not found in default locations."
  echo "Will auto-generate fingerprint_cache.npz from finetuning.hdf5"
fi

if [ ! -f "$SRC_DIR/create_peak_embeddings.py" ] || [ ! -f "$SRC_DIR/frozen_allpeaks_baselines.py" ] || [ ! -f "$SRC_DIR/frozen_allpeaks_inference.py" ] || [ ! -f "$SRC_DIR/create_fingerprint_cache.py" ]; then
  echo "Error: required source scripts not found under $SRC_DIR"
  exit 1
fi

# ------------------------------------------------------------------
# Stage large input data to scratch
# ------------------------------------------------------------------
echo "Staging datasets to scratch..."
cp "$INPUT_FINETUNING_HDF5" "$SCRATCH_DATA_DIR/finetuning.hdf5"
cp "$INPUT_FINETUNING_WITH_SSL_HDF5" "$SCRATCH_DATA_DIR/finetuning_with_ssl_embeddings.hdf5"
cp "$INPUT_PROBING_TEST" "$SCRATCH_DATA_DIR/probing_test.parquet"

if [ -n "${INPUT_FP_CACHE:-}" ]; then
  cp "$INPUT_FP_CACHE" "$SCRATCH_DATA_DIR/fingerprint_cache.npz"
else
  echo "Generating fingerprint cache on scratch..."
  srun --export=ALL --preserve-env python "$SRC_DIR/create_fingerprint_cache.py" \
    --finetuning-hdf5 "$SCRATCH_DATA_DIR/finetuning.hdf5" \
    --output "$SCRATCH_DATA_DIR/fingerprint_cache.npz"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Step 1: Extract peak embeddings on GPU
# ------------------------------------------------------------------
EXTRACT_CMD=(
  python "$SRC_DIR/create_peak_embeddings.py"
  --hdf5 "$SCRATCH_DATA_DIR/finetuning_with_ssl_embeddings.hdf5"
  --batch-size "$EMB_BATCH_SIZE"
  --dtype "$EMB_DTYPE"
  --folds "$EMB_FOLDS"
  --device cuda
  --write-peak-mask
)

if [ "$OVERWRITE_PEAKS" = "1" ]; then
  EXTRACT_CMD+=(--overwrite)
fi

echo "Running peak extraction..."
srun --export=ALL --preserve-env "${EXTRACT_CMD[@]}"

# ------------------------------------------------------------------
# Step 2: Train 6 frozen-allpeaks heads
# ------------------------------------------------------------------
train_one () {
  local fp_kind="$1"
  local loss_kind="$2"
  local run_tag="$3"

  local CMD=(
    python "$SRC_DIR/frozen_allpeaks_baselines.py"
    --project-root "$REPO_ROOT"
    --embedding-hdf5 "$SCRATCH_DATA_DIR/finetuning_with_ssl_embeddings.hdf5"
    --finetuning-hdf5 "$SCRATCH_DATA_DIR/finetuning.hdf5"
    --fingerprint-cache "$SCRATCH_DATA_DIR/fingerprint_cache.npz"
    --fp-kind "$fp_kind"
    --loss-kind "$loss_kind"
    --run-tag "$run_tag"
    --batch-size "$HEAD_BATCH_SIZE"
    --max-epochs "$MAX_EPOCHS"
    --early-stop-patience "$EARLY_STOP_PATIENCE"
    --num-workers 0
    --device cuda
  )

  if [ "$USE_HDF5_PEAK_MASK" = "1" ]; then
    CMD+=(--use-hdf5-peak-mask)
  fi

  echo "Training $run_tag ..."
  srun --export=ALL --preserve-env "${CMD[@]}"
}

train_one morgan_2048 bce_logits morgan_2048_bce_frozen_allpeaks
train_one maccs_166 bce_logits maccs_166_bce_frozen_allpeaks
train_one map4_2048 bce_logits map4_2048_bce_frozen_allpeaks
train_one morgan_2048 cos morgan_2048_cos_frozen_allpeaks
train_one maccs_166 cos maccs_166_cos_frozen_allpeaks
train_one map4_2048 cos map4_2048_cos_frozen_allpeaks

# ------------------------------------------------------------------
# Step 3: Run inference and save y_pred/y_true caches to model_runs
# ------------------------------------------------------------------
INFER_CMD=(
  python "$SRC_DIR/frozen_allpeaks_inference.py"
  --project-root "$REPO_ROOT"
  --embedding-hdf5 "$SCRATCH_DATA_DIR/finetuning_with_ssl_embeddings.hdf5"
  --finetuning-hdf5 "$SCRATCH_DATA_DIR/finetuning.hdf5"
  --probing-test "$SCRATCH_DATA_DIR/probing_test.parquet"
  --frozen-allpeaks-root "$PERSIST_RESULTS_FROZEN_ALLPEAKS"
  --model-runs-root "$PERSIST_RESULTS_MODEL_RUNS"
  --batch-size "$INFER_BATCH_SIZE"
  --device cuda
)

if [ -n "$RUN_TAGS" ]; then
  INFER_CMD+=(--run-tags "$RUN_TAGS")
fi

echo "Running frozen-allpeaks inference..."
srun --export=ALL --preserve-env "${INFER_CMD[@]}"

# ------------------------------------------------------------------
# Step 4: Optional compact output summary and cleanup
# ------------------------------------------------------------------
echo ""
echo "Pipeline finished. Key outputs:"
echo "  - Checkpoints/histories: $PERSIST_RESULTS_FROZEN_ALLPEAKS/<run_tag>/"
echo "  - Cached preds/targets : $PERSIST_RESULTS_MODEL_RUNS/<run_tag>/axis2_artifacts/"

echo "Cleaning scratch: $SCRATCH_ROOT"
rm -rf "$SCRATCH_ROOT"

echo "Done."

# Example usage:
# sbatch dreams-thesis-wa/scripts/h100_frozen_allpeaks_pipeline.sh
#
# Run subset only:
# sbatch --export=RUN_TAGS=morgan_2048_bce_frozen_allpeaks,morgan_2048_cos_frozen_allpeaks dreams-thesis-wa/scripts/h100_frozen_allpeaks_pipeline.sh
#
# Keep train/val only in extraction and float16 storage:
# sbatch --export=EMB_DTYPE=float16,EMB_FOLDS=train,val dreams-thesis-wa/scripts/h100_frozen_allpeaks_pipeline.sh
