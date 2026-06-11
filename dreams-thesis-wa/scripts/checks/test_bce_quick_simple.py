#!/usr/bin/env python3
"""
Quick test: Run BCE loss on 100 samples for 1 epoch.
Usage: python dreams-thesis-wa/scripts/checks/test_bce_quick_simple.py --loss bce_logits --samples 100 --epochs 1
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Quick BCE loss test")
    parser.add_argument('--loss', choices=['cos', 'bce_logits', 'bce'], 
                       default='bce_logits', help='Loss function to test')
    parser.add_argument('--samples', type=int, default=100, 
                       help='Number of samples to use')
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of epochs to run')
    parser.add_argument('--device', choices=['cpu', 'gpu'], 
                       default='gpu' if __import__('torch').cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Paths
    dataset_path = Path.home() / "DreaMS/dreams-thesis-wa/data/processed/finetuning.hdf5"
    pretrained_path = Path(f"{Path.home()}/DreaMS/pretrained/ssl_model.ckpt") if not Path(f"{Path.home()}/DreaMS/pretrained/ssl_model.ckpt").exists() else None
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run: python dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py")
        return 1
    
    print("=" * 70)
    print(f"Quick BCE Test: {args.samples} samples, {args.epochs} epoch(s), loss={args.loss}")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Device: {args.device}")
    print()
    
    # Build training command
    cmd = [
        sys.executable, "dreams/training/train.py",
        "--run_name", f"test_{args.loss}_quick",
        "--project_name", "DreaMS_BCE_Test",
        "--job_key", f"test_{args.loss}",
        "--dataset_pth", str(dataset_path),
        "--dformat", "B",
        "--batch_size", "64",
        "--n_samples", str(args.samples),
        "--max_epochs", str(args.epochs),
        "--num_workers_data", "4",
        "--fingerprint_type", "morgan_2048",
        "--fp_loss", args.loss,
        "--seed", "3407",
        "--num_devices", "1" if args.device == 'gpu' else "1",
        "--no_wandb"
    ]
    
    # Add pre-trained if available
    if pretrained_path and pretrained_path.exists():
        cmd.extend(["--pretrained_pth", str(pretrained_path)])
    
    print(f"Running: {' '.join(cmd[4:])}")
    print()
    
    import subprocess
    result = subprocess.run(cmd, cwd=Path.home() / "DreaMS")
    
    print()
    print("=" * 70)
    if result.returncode == 0:
        print(f"✅ Test passed: {args.loss} loss")
    else:
        print(f"❌ Test failed: {args.loss} loss")
    print("=" * 70)
    print()
    
    if result.returncode == 0:
        print("Next steps:")
        print("  1. Run other loss variants:")
        print("     python dreams-thesis-wa/scripts/checks/test_bce_quick_simple.py --loss cos")
        print("     python dreams-thesis-wa/scripts/checks/test_bce_quick_simple.py --loss bce_logits --samples 500 --epochs 2")
        print()
        print("  2. Submit full SLURM runs:")
        print("     sbatch fine_tune_test.sh         # Cosine baseline")
        print("     sbatch fine_tune_test_bce.sh     # BCE comparison")
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
