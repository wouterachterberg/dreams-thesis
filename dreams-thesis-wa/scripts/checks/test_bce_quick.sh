#!/bin/bash
# Quick test to validate BCE loss implementation on small sample.
# Runs 1 epoch with only 100 samples - should complete in <5 minutes.
# Use this to verify BCE works before submitting full SLURM runs.

set -e  # Exit on error

# Configuration
DATASET_PATH="${HOME}/DreaMS/dreams-thesis-wa/data/processed/finetuning.hdf5"
PRETRAINED_PATH="${PRETRAINED}/ssl_model.ckpt"
CHECKPOINTS_DIR="/tmp/dreams_bce_test_$$"

echo "======================================"
echo "Quick BCE Loss Implementation Test"
echo "======================================"
echo ""

# Verify dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "   Run: python dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py"
    exit 1
fi
echo "✓ Dataset found: $DATASET_PATH"

# Verify pre-trained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Pre-trained model not found: $PRETRAINED_PATH"
    echo "   Will attempt to download or use local fallback"
fi

# Setup checkpoint directory
mkdir -p "$CHECKPOINTS_DIR"
echo "✓ Checkpoints will be saved to: $CHECKPOINTS_DIR"
echo ""

# Test 1: Quick cosine baseline (reference)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Cosine Loss (baseline, 100 samples, 1 epoch)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python dreams/training/train.py \
  --run_name "test_cosine_quick" \
  --project_name "DreaMS_BCE_Test" \
  --job_key "test_cosine" \
  --dataset_pth "$DATASET_PATH" \
  --dformat B \
  --batch_size 64 \
  --n_samples 100 \
  --max_epochs 1 \
  --num_workers_data 4 \
  --pretrained_pth "$PRETRAINED_PATH" \
  --fingerprint_type morgan_2048 \
  --fp_loss cos \
  --seed 3407 \
  --num_devices 1 \
  --checkpoints_dir "$CHECKPOINTS_DIR/cos" \
  --no_wandb

if [ $? -eq 0 ]; then
    echo "✓ Cosine test passed"
else
    echo "❌ Cosine test failed"
    exit 1
fi
echo ""

# Test 2: Quick BCE with logits (new feature)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: BCE with Logits (new, 100 samples, 1 epoch)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python dreams/training/train.py \
  --run_name "test_bce_quick" \
  --project_name "DreaMS_BCE_Test" \
  --job_key "test_bce" \
  --dataset_pth "$DATASET_PATH" \
  --dformat B \
  --batch_size 64 \
  --n_samples 100 \
  --max_epochs 1 \
  --num_workers_data 4 \
  --pretrained_pth "$PRETRAINED_PATH" \
  --fingerprint_type morgan_2048 \
  --fp_loss bce_logits \
  --seed 3407 \
  --num_devices 1 \
  --checkpoints_dir "$CHECKPOINTS_DIR/bce" \
  --no_wandb

if [ $? -eq 0 ]; then
    echo "✓ BCE test passed"
else
    echo "❌ BCE test failed"
    exit 1
fi
echo ""

# Test 3: BCE with pos_weight (sparse handling)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: BCE with pos_weight (sparse, 100 samples, 1 epoch)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python dreams/training/train.py \
  --run_name "test_bce_posweight_quick" \
  --project_name "DreaMS_BCE_Test" \
  --job_key "test_bce_pos" \
  --dataset_pth "$DATASET_PATH" \
  --dformat B \
  --batch_size 64 \
  --n_samples 100 \
  --max_epochs 1 \
  --num_workers_data 4 \
  --pretrained_pth "$PRETRAINED_PATH" \
  --fingerprint_type morgan_2048 \
  --fp_loss bce_logits \
  --fp_pos_weight 44.0 \
  --seed 3407 \
  --num_devices 1 \
  --checkpoints_dir "$CHECKPOINTS_DIR/bce_pos" \
  --no_wandb

if [ $? -eq 0 ]; then
    echo "✓ BCE with pos_weight test passed"
else
    echo "❌ BCE with pos_weight test failed"
    exit 1
fi
echo ""

# Cleanup
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "Remove temporary checkpoints? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$CHECKPOINTS_DIR"
    echo "✓ Cleanup complete"
else
    echo "Checkpoints preserved at: $CHECKPOINTS_DIR"
fi

echo ""
echo "======================================"
echo "✅ All tests passed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Review loss values above - should decrease over epochs"
echo "  2. Check that BCE loss is NOT applied before passing to loss function"
echo "  3. Submit full runs:"
echo "     sbatch fine_tune_test.sh                 # Cosine baseline"
echo "     sbatch fine_tune_test_bce.sh             # BCE comparison"
echo ""
