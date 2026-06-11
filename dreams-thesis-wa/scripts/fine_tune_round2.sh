#!/bin/bash
#
# Round 2 fine-tuning for the BCE fingerprint conditions with fp64 and a
# step-wise OneCycleLR schedule.
#
# Default submission runs one condition:
#   sbatch dreams-thesis-wa/scripts/fine_tune_round2.sh
#
# Run a different condition:
#   sbatch --export=CONDITION_KEYS=morgan dreams-thesis-wa/scripts/fine_tune_round2.sh
#
# Run all three sequentially:
#   sbatch --export=CONDITION_KEYS=maccs,morgan,map4 dreams-thesis-wa/scripts/fine_tune_round2.sh
#
# Optional per-condition BCE pos_weight overrides:
#   MACCS_FP_POS_WEIGHT, MORGAN_FP_POS_WEIGHT, MAP4_FP_POS_WEIGHT
#
# NOTE: MAP4 requires the `map4` package on Snellius.
#
#SBATCH --job-name=DreaMS_ft_round2
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

module load 2024
module load Miniconda3/24.7.1-0

eval "$(conda shell.bash hook)"
conda activate dreams

$(python -c "from dreams.definitions import export; export()")

FP_LOSS=${FP_LOSS:-bce_logits}
LOSS_LABEL=${LOSS_LABEL:-bce}
PROJECT_NAME=${PROJECT_NAME:-DreaMS_Finetuning}
WANDB_PROJECT=${WANDB_PROJECT:-$PROJECT_NAME}
REPO_ROOT="${REPO_ROOT:-${DREAMS_REPO_ROOT:-$HOME/DreaMS}}"
SCRATCH_BASE="${SCRATCH_BASE:-${DREAMS_SCRATCH_BASE:-/scratch-shared/$USER}}"
RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-_r2}

MACCS_OBJECTIVE=${MACCS_OBJECTIVE:-fp_maccs_166}
MACCS_RUN_NAME=${MACCS_RUN_NAME:-${MACCS_OBJECTIVE}_${LOSS_LABEL}${RUN_NAME_SUFFIX}}
MACCS_FP_POS_WEIGHT=${MACCS_FP_POS_WEIGHT:-}

MORGAN_OBJECTIVE=${MORGAN_OBJECTIVE:-fp_morgan_2048}
MORGAN_RUN_NAME=${MORGAN_RUN_NAME:-${MORGAN_OBJECTIVE}_${LOSS_LABEL}${RUN_NAME_SUFFIX}}
MORGAN_FP_POS_WEIGHT=${MORGAN_FP_POS_WEIGHT:-}

MAP4_OBJECTIVE=${MAP4_OBJECTIVE:-fp_map4_2048}
MAP4_RUN_NAME=${MAP4_RUN_NAME:-${MAP4_OBJECTIVE}_${LOSS_LABEL}${RUN_NAME_SUFFIX}}
MAP4_FP_POS_WEIGHT=${MAP4_FP_POS_WEIGHT:-}

CONDITION_KEYS=${CONDITION_KEYS:-maccs}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE:-256}
DEFAULT_ACCUMULATE_GRAD_BATCHES=${DEFAULT_ACCUMULATE_GRAD_BATCHES:-1}

HIGH_MEM_BATCH_SIZE=${HIGH_MEM_BATCH_SIZE:-128}
HIGH_MEM_ACCUMULATE_GRAD_BATCHES=${HIGH_MEM_ACCUMULATE_GRAD_BATCHES:-2}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.wandb_secrets" ]; then
    source "$SCRIPT_DIR/.wandb_secrets"
    echo "Loaded WandB credentials from .wandb_secrets"
elif [ -f "$HOME/.wandb_secrets" ]; then
    source "$HOME/.wandb_secrets"
    echo "Loaded WandB credentials from ~/.wandb_secrets"
elif [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: No WandB credentials found."
    echo "Create .wandb_secrets from .wandb_secrets.template"
    echo "Or set WANDB_API_KEY as an environment variable"
fi

resolve_condition() {
    local condition_key="$1"

    case "$condition_key" in
        maccs)
            FP_OBJECTIVE="$MACCS_OBJECTIVE"
            RUN_NAME="$MACCS_RUN_NAME"
            FP_POS_WEIGHT="$MACCS_FP_POS_WEIGHT"
            BATCH_SIZE="$DEFAULT_BATCH_SIZE"
            ACCUMULATE_GRAD_BATCHES="$DEFAULT_ACCUMULATE_GRAD_BATCHES"
            ;;
        morgan)
            FP_OBJECTIVE="$MORGAN_OBJECTIVE"
            RUN_NAME="$MORGAN_RUN_NAME"
            FP_POS_WEIGHT="$MORGAN_FP_POS_WEIGHT"
            BATCH_SIZE="$HIGH_MEM_BATCH_SIZE"
            ACCUMULATE_GRAD_BATCHES="$HIGH_MEM_ACCUMULATE_GRAD_BATCHES"
            ;;
        map4)
            FP_OBJECTIVE="$MAP4_OBJECTIVE"
            RUN_NAME="$MAP4_RUN_NAME"
            FP_POS_WEIGHT="$MAP4_FP_POS_WEIGHT"
            BATCH_SIZE="$HIGH_MEM_BATCH_SIZE"
            ACCUMULATE_GRAD_BATCHES="$HIGH_MEM_ACCUMULATE_GRAD_BATCHES"
            ;;
        *)
            echo "Unknown CONDITION_KEYS entry: $condition_key"
            echo "Expected one of: maccs, morgan, map4"
            exit 2
            ;;
    esac
}

run_condition() {
    local condition_key="$1"
    local -a FP_POS_WEIGHT_ARGS
    local -a TRAIN_ARGS
    local SCRATCH_DIR
    local SCRATCH_DATASET
    local SCRATCH_PRETRAINED
    local HOME_RUN_DIR
    local CHECKPOINT_DIR

    resolve_condition "$condition_key"

    SCRATCH_DIR="$SCRATCH_BASE/dreams_finetune_${RUN_NAME}"

    echo "Setting up scratch-shared workspace: $SCRATCH_DIR"
    mkdir -p "$SCRATCH_DIR"

    echo "Copying dataset to scratch..."
    cp "$REPO_ROOT/dreams-thesis-wa/data/processed/finetuning.hdf5" "$SCRATCH_DIR/"
    echo "Copying pre-trained model to scratch..."
    cp "${PRETRAINED}/ssl_model.ckpt" "$SCRATCH_DIR/"

    SCRATCH_DATASET="$SCRATCH_DIR/finetuning.hdf5"
    SCRATCH_PRETRAINED="$SCRATCH_DIR/ssl_model.ckpt"
    echo "Scratch setup complete"

    HOME_RUN_DIR="$REPO_ROOT/dreams-thesis-wa/results/finetuning/$RUN_NAME"
    mkdir -p "$HOME_RUN_DIR"

    if [ -n "$FP_POS_WEIGHT" ]; then
        FP_POS_WEIGHT_ARGS=(--fp_pos_weight "$FP_POS_WEIGHT")
    else
        FP_POS_WEIGHT_ARGS=()
    fi

    if [ ! -f "$SCRATCH_DATASET" ]; then
        echo "Error: Dataset not found at $SCRATCH_DATASET"
        echo "Please run: python dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py"
        exit 1
    fi

    if [ ! -f "$SCRATCH_PRETRAINED" ]; then
        echo "Error: Pre-trained model not found. Check \$PRETRAINED."
        exit 1
    fi

    echo ""
    echo "=================================="
    echo "MassSpecGym Fine-Tuning Round 2"
    echo "=================================="
    echo "  Condition   : $condition_key"
    echo "  Objective   : $FP_OBJECTIVE"
    echo "  Loss        : $FP_LOSS"
    echo "  Pos weight  : ${FP_POS_WEIGHT:-none}"
    echo "  Batch size  : $BATCH_SIZE"
    echo "  Grad accum  : $ACCUMULATE_GRAD_BATCHES"
    echo "  CUDA alloc  : $PYTORCH_CUDA_ALLOC_CONF"
    echo "  Project     : $WANDB_PROJECT"
    echo "  Run name    : $RUN_NAME"
    echo "  Dataset     : $SCRATCH_DATASET"
    echo "  Pre-trained : $SCRATCH_PRETRAINED"
    echo "  Output dir  : $HOME_RUN_DIR"
    echo "=================================="
    echo ""

    TRAIN_ARGS=(
        --project_name "$WANDB_PROJECT"
        --job_key "$RUN_NAME"
        --run_name "$RUN_NAME"
        --train_objective "$FP_OBJECTIVE"
        --fp_loss "$FP_LOSS"
        --train_regime fine-tuning
        --dataset_pth "$SCRATCH_DATASET"
        --dformat A
        --model DreaMS
        --lr 1.5e-5
        --batch_size "$BATCH_SIZE"
        --prec_intens 1.1
        --num_devices 4
        --max_epochs 103
        --log_every_n_steps 5
        --head_depth 1
        --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES"
        --seed 3407
        --train_precision 64
        --use_lr_schedule
        --pre_trained_pth "$SCRATCH_PRETRAINED"
        --val_check_interval 0.5
        --max_peaks_n 100
        --save_top_k 3
        --num_workers_data 32
        --early_stopping_patience 20
    )

    if [ -n "$WANDB_PROJECT" ] && [ "$WANDB_PROJECT" != "your-wandb-project-name" ]; then
        TRAIN_ARGS+=(--wandb_entity_name "$WANDB_ENTITY")
        echo "WandB logging enabled"
    else
        TRAIN_ARGS+=(--no_wandb)
        echo "WandB logging disabled"
    fi

    if [ "${#FP_POS_WEIGHT_ARGS[@]}" -gt 0 ]; then
        TRAIN_ARGS+=("${FP_POS_WEIGHT_ARGS[@]}")
    fi

    echo ""

    cd "$SCRATCH_DIR" || exit 3
    export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

    srun --export=ALL --preserve-env python3 "$REPO_ROOT/dreams/training/train.py" \
        "${TRAIN_ARGS[@]}"

    CHECKPOINT_DIR="$SCRATCH_DIR/$WANDB_PROJECT/$RUN_NAME"
    echo ""
    echo "Collecting output from $CHECKPOINT_DIR ..."
    if [ -d "$CHECKPOINT_DIR" ]; then
        cp -r "$CHECKPOINT_DIR"/* "$HOME_RUN_DIR/"
        echo "Checkpoints copied to: $HOME_RUN_DIR"

        cd "$SCRATCH_DIR" || exit 4
        zip -r "${RUN_NAME}_checkpoints.zip" "$WANDB_PROJECT/$RUN_NAME/"
        mv "${RUN_NAME}_checkpoints.zip" "$HOME_RUN_DIR/"
        echo "Archive created: $HOME_RUN_DIR/${RUN_NAME}_checkpoints.zip"
    else
        echo "Checkpoint directory not found at $CHECKPOINT_DIR"
        echo "Listing scratch contents for debugging:"
        find "$SCRATCH_DIR" -maxdepth 3 -type d
    fi

    echo "Cleaning up scratch..."
    rm -rf "$SCRATCH_DIR"
    echo "Scratch cleaned up"

    echo ""
    echo "=================================="
    echo "Round 2 fine-tuning complete for $RUN_NAME"
    echo "All output saved to: $HOME_RUN_DIR"
    echo "=================================="
    echo ""
}

IFS=',' read -r -a SELECTED_CONDITIONS <<< "$CONDITION_KEYS"
for condition_key in "${SELECTED_CONDITIONS[@]}"; do
    condition_key="${condition_key// /}"
    if [ -n "$condition_key" ]; then
        run_condition "$condition_key"
    fi
done
