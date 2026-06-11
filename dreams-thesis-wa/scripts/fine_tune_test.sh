#!/bin/bash
#
# BASE FINE-TUNING SCRIPT — all variant scripts (maccs, map4, bce, etc.)
# set environment variables and then `exec bash` into this script.
#
# Configurable via environment variables:
#   FP_OBJECTIVE   — fingerprint target  (default: fp_morgan_2048)
#   FP_LOSS        — loss function        (default: cos)
#   PROJECT_NAME   — WandB project name   (default: derived from FP_OBJECTIVE)
#   FP_POS_WEIGHT  — BCE positive-class weight (optional, only for bce_logits)
#   RUN_NAME       — override run name    (default: auto-generated)
#
#SBATCH --job-name=DreaMS_fine-tuning
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# ── Modules & conda ──────────────────────────────────────────────────
module load 2024
module load Miniconda3/24.7.1-0

eval "$(conda shell.bash hook)"
conda activate dreams

# Export project definitions (PRETRAINED, MERGED_DATASETS, etc.)
$(python -c "from dreams.definitions import export; export()")

# ── Configuration ────────────────────────────────────────────────────
FP_OBJECTIVE=${FP_OBJECTIVE:-fp_morgan_2048}
FP_LOSS=${FP_LOSS:-cos}
PROJECT_NAME=${PROJECT_NAME:-DreaMS_Finetuning}
WANDB_PROJECT=${WANDB_PROJECT:-$PROJECT_NAME}
REPO_ROOT="${REPO_ROOT:-${DREAMS_REPO_ROOT:-$HOME/DreaMS}}"
SCRATCH_BASE="${SCRATCH_BASE:-${DREAMS_SCRATCH_BASE:-/scratch-shared/$USER}}"

# Unique, descriptive run name: <objective>_<loss>_<timestamp>
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-${FP_OBJECTIVE}_${FP_LOSS}_${RUN_TIMESTAMP}}

# ── Scratch workspace (fast NVMe on compute node) ───────────────────
SCRATCH_DIR="$SCRATCH_BASE/dreams_finetune_${RUN_NAME}"

echo "Setting up scratch-shared workspace: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

echo "Copying dataset to scratch..."
cp "$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5" "$SCRATCH_DIR/"
echo "Copying pre-trained model to scratch..."
cp "${PRETRAINED}/ssl_model.ckpt" "$SCRATCH_DIR/"

SCRATCH_DATASET="$SCRATCH_DIR/finetuning.hdf5"
SCRATCH_PRETRAINED="$SCRATCH_DIR/ssl_model.ckpt"
echo "✅ Scratch setup complete"

# ── Per-run output directory on $HOME (persistent) ──────────────────
# Structure: results/finetuning/<RUN_NAME>/
#   ├── checkpoints/ (*.ckpt files)
#   └── <RUN_NAME>_checkpoints.zip
HOME_RUN_DIR="$REPO_ROOT/dreams-thesis-wa/results/finetuning/$RUN_NAME"
mkdir -p "$HOME_RUN_DIR"

# ── WandB credentials ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.wandb_secrets" ]; then
    source "$SCRIPT_DIR/.wandb_secrets"
    echo "✅ Loaded WandB credentials from .wandb_secrets"
elif [ -f "$HOME/.wandb_secrets" ]; then
    source "$HOME/.wandb_secrets"
    echo "✅ Loaded WandB credentials from ~/.wandb_secrets"
elif [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: No WandB credentials found!"
    echo "   Create .wandb_secrets from .wandb_secrets.template"
    echo "   Or set WANDB_API_KEY environment variable"
fi

# ── Optional BCE pos_weight ──────────────────────────────────────────
FP_POS_WEIGHT=${FP_POS_WEIGHT:-}
FP_POS_WEIGHT_ARGS=""
if [ -n "$FP_POS_WEIGHT" ]; then
    FP_POS_WEIGHT_ARGS="--fp_pos_weight $FP_POS_WEIGHT"
fi

# ── Validation ───────────────────────────────────────────────────────
if [ ! -f "$SCRATCH_DATASET" ]; then
    echo "Error: Dataset not found at $SCRATCH_DATASET"
    echo "Please run: python dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py"
    exit 1
fi

if [ ! -f "$SCRATCH_PRETRAINED" ]; then
    echo "Error: Pre-trained model not found. Check \$PRETRAINED variable."
    exit 1
fi

echo ""
echo "=================================="
echo "MassSpecGym Fine-Tuning"
echo "=================================="
echo "  Objective   : $FP_OBJECTIVE"
echo "  Loss        : $FP_LOSS"
echo "  Pos weight  : ${FP_POS_WEIGHT:-none}"
echo "  Project     : $WANDB_PROJECT"
echo "  Run name    : $RUN_NAME"
echo "  Dataset     : $SCRATCH_DATASET"
echo "  Pre-trained : $SCRATCH_PRETRAINED"
echo "  Output dir  : $HOME_RUN_DIR"
echo "=================================="
echo ""

# ── WandB arguments ──────────────────────────────────────────────────
WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ] && [ "$WANDB_PROJECT" != "your-wandb-project-name" ]; then
    WANDB_ARGS="--project_name $WANDB_PROJECT --wandb_entity_name $WANDB_ENTITY"
    echo "WandB logging enabled"
else
    WANDB_ARGS="--no_wandb"
    echo "WandB logging disabled"
fi
echo ""

# ── Run training ─────────────────────────────────────────────────────
# cd to scratch so Lightning writes checkpoints to fast storage
cd "$SCRATCH_DIR" || exit 3
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

srun --export=ALL --preserve-env python3 "$REPO_ROOT/dreams/training/train.py" \
 $WANDB_ARGS \
 --job_key "$RUN_NAME" \
 --run_name "$RUN_NAME" \
 --train_objective "$FP_OBJECTIVE" \
 --fp_loss "$FP_LOSS" \
 $FP_POS_WEIGHT_ARGS \
 --train_regime fine-tuning \
 --dataset_pth "$SCRATCH_DATASET" \
 --dformat A \
 --model DreaMS \
 --lr 1.5e-5 \ ##evt scheduler
 --batch_size 256 \
 --prec_intens 1.1 \ 
 --num_devices 4 \
 --max_epochs 103 \
 --log_every_n_steps 5 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \ ##evt
 --pre_trained_pth "$SCRATCH_PRETRAINED" \
 --val_check_interval 0.5 \
 --max_peaks_n 100 \
 --save_top_k 3 \
 --num_workers_data 32 \
 --early_stopping_patience 20

# ── Copy checkpoints from scratch → $HOME per-run directory ─────────
# Lightning saves to: <cwd>/<project_name>/<job_key>/
CHECKPOINT_DIR="$SCRATCH_DIR/$WANDB_PROJECT/$RUN_NAME"
echo ""
echo "Collecting output from $CHECKPOINT_DIR ..."
if [ -d "$CHECKPOINT_DIR" ]; then
    # Copy individual checkpoint files
    cp -r "$CHECKPOINT_DIR"/* "$HOME_RUN_DIR/"
    echo "✅ Checkpoints copied to: $HOME_RUN_DIR"

    # Also create a zip archive for easy transfer
    cd "$SCRATCH_DIR"
    zip -r "${RUN_NAME}_checkpoints.zip" "$WANDB_PROJECT/$RUN_NAME/"
    mv "${RUN_NAME}_checkpoints.zip" "$HOME_RUN_DIR/"
    echo "✅ Archive created: $HOME_RUN_DIR/${RUN_NAME}_checkpoints.zip"
else
    echo "⚠️  Checkpoint directory not found at $CHECKPOINT_DIR"
    echo "   Listing scratch contents for debugging:"
    find "$SCRATCH_DIR" -maxdepth 3 -type d
fi

# ── Cleanup ──────────────────────────────────────────────────────────
echo "Cleaning up scratch..."
rm -rf "$SCRATCH_DIR"
echo "✅ Scratch cleaned up"

echo ""
echo "=================================="
echo "Fine-tuning complete!"
echo "All output saved to: $HOME_RUN_DIR"
echo "=================================="
