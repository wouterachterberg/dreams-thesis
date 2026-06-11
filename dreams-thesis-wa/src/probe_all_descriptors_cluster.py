#!/usr/bin/env python3
"""
Standalone script for probing ALL RDKit descriptors on GPU cluster.
This script trains both Linear and MLP probes for ~200 descriptors.

GPU OPTIMIZATIONS:
    - ProcessPoolExecutor for true multi-GPU parallelism (not threads)
    - Increased batch size to 4096 (from 1024) for better GPU utilization
    - Multi-worker DataLoaders with pin_memory for faster CPU->GPU transfer
    - Non-blocking GPU transfers for overlapping computation
    - Batched inference for memory efficiency
    - MultiTaskProbe option for training multiple descriptors simultaneously

Usage:
    python probe_all_descriptors_cluster.py  # Auto-detects 1 free GPU
    python probe_all_descriptors_cluster.py --device cuda:0  # Use specific GPU
    python probe_all_descriptors_cluster.py --devices cuda:0,cuda:1  # Use multiple GPUs (if needed)

Outputs:
    - results/all_descriptors_probing_results_linear.pkl
    - results/all_descriptors_probing_results_mlp.pkl
    - results/cluster_run_summary.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from datetime import datetime
import sys
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import subprocess

# =============================================================================
# GPU Optimization Settings
# =============================================================================

# Enable TF32 for faster computation on A30 GPUs (Ampere architecture)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN autotuner for optimal convolution algorithms
    torch.backends.cudnn.benchmark = True
    # Disable CUDA memory caching allocator for more predictable memory usage
    # Uncomment if you experience OOM errors:
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# =============================================================================
# Configuration
# =============================================================================

# Paths (adjust these for your cluster environment)
DATA_PATH = Path('../data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet')
DESCRIPTORS_PATH = Path('../data/processed/massspecgym_complete/all_rdkit_descriptors.parquet')
RESULTS_DIR = Path('../results')

# Model parameters
BATCH_SIZE = 1024  # Significantly increased to utilize GPU memory better
LEARNING_RATE = 0.001
EPOCHS = 30
DROPOUT = 0.2
NUM_WORKERS_LINEAR = 0  # Set to 0 to avoid file handle issues with ProcessPoolExecutor
NUM_WORKERS_MLP = 0  # Set to 0 to avoid file handle issues with ProcessPoolExecutor
PIN_MEMORY = True  # Pin memory for faster GPU transfer

# =============================================================================
# Model Definitions
# ==============================gi===============================================

class LinearProbe(nn.Module):
    """Simple linear regression probe: 1024→1"""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """3-layer MLP probe with ReLU and Dropout: 1024→256→128→1"""
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class MultiTaskProbe(nn.Module):
    """Multi-task probe that predicts multiple descriptors simultaneously.
    This allows better GPU utilization by processing multiple tasks in parallel."""
    def __init__(self, input_dim, num_tasks, hidden_dim=256, dropout=0.2):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Task-specific heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_tasks)])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)  # (batch_size, num_tasks)


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings and targets"""
    def __init__(self, embeddings, targets):
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


# =============================================================================
# Training Function
# =============================================================================

def train_probe(X_train, y_train, X_test, y_test, model_type='mlp', 
                epochs=30, lr=0.001, device='cuda:0', verbose=False, num_workers=0):
    """
    Train probe and evaluate.
    
    Args:
        X_train: Training embeddings (N, 1024)
        y_train: Training targets (N,)
        X_test: Test embeddings (M, 1024)
        y_test: Test targets (M,)
        model_type: 'linear' or 'mlp'
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda:0', 'cuda:1', or 'cpu'
        verbose: Print training progress
        num_workers: Number of DataLoader workers (0 for MLP, 4 for Linear)
    
    Returns:
        dict with r2, mae, model, scaler
    """
    # Standardize targets
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train_scaled)
    
    # Optimized DataLoader with multiple workers and pin_memory
    is_cuda = device.startswith('cuda')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers if is_cuda else 0,
        pin_memory=PIN_MEMORY if is_cuda else False,
        persistent_workers=True if is_cuda and num_workers > 0 else False
    )
    
    # Create model
    if model_type == 'linear':
        model = LinearProbe(input_dim=X_train.shape[1])
    else:
        model = MLPProbe(input_dim=X_train.shape[1], dropout=DROPOUT)
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for embeddings, targets in train_loader:
            embeddings, targets = embeddings.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluate in batches to handle large test sets efficiently
    model.eval()
    all_preds_scaled = []
    test_dataset = EmbeddingDataset(X_test, y_test_scaled)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=num_workers if is_cuda else 0,
        pin_memory=PIN_MEMORY if is_cuda else False
    )
    
    with torch.no_grad():
        for embeddings, _ in test_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            preds = model(embeddings).cpu().numpy().flatten()
            all_preds_scaled.extend(preds)
    
    preds_scaled = np.array(all_preds_scaled)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        'r2': r2,
        'mae': mae,
        'model': model.cpu(),  # Move back to CPU for saving
        'scaler': scaler
    }


# =============================================================================
# GPU Detection Utility
# =============================================================================

def get_free_gpus(max_gpus=2, memory_threshold=1000):
    """
    Detect free GPUs based on memory usage.
    
    Args:
        max_gpus: Maximum number of GPUs to return (default: 2)
        memory_threshold: MB of memory used to consider GPU as "in use" (default: 1000 MB)
    
    Returns:
        List of free GPU indices (e.g., [0, 1] or [2, 3])
    """
    try:
        # Run nvidia-smi to get GPU memory usage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                gpu_id, memory_used = line.split(',')
                gpu_info.append({
                    'id': int(gpu_id.strip()),
                    'memory_used': float(memory_used.strip())
                })
        
        # Sort by GPU ID and filter free GPUs
        free_gpus = [
            gpu['id'] for gpu in sorted(gpu_info, key=lambda x: x['id'])
            if gpu['memory_used'] < memory_threshold
        ]
        
        if not free_gpus:
            print("⚠️  No free GPUs found! All GPUs are in use.")
            return None
        
        # Return up to max_gpus sequential GPUs
        selected_gpus = free_gpus[:max_gpus]
        
        print(f"🔍 GPU Status:")
        for gpu in gpu_info:
            status = "✅ FREE" if gpu['id'] in selected_gpus else f"❌ IN USE ({gpu['memory_used']:.0f} MB)"
            selected = "← SELECTED" if gpu['id'] in selected_gpus else ""
            print(f"   GPU {gpu['id']}: {status} {selected}")
        
        return selected_gpus
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  Could not detect GPUs: {e}")
        print("   Falling back to manual detection...")
        return None


# =============================================================================
# Main Probing Function
# =============================================================================

def probe_single_descriptor(desc, df_train, df_test, X_train, X_test, model_type, device):
    """
    Probe a single descriptor on a specific device.
    
    Returns:
        dict with results or None if failed
    """
    # Get targets (drop NaN)
    train_mask = df_train[desc].notna()
    test_mask = df_test[desc].notna()
    
    if train_mask.sum() < 100 or test_mask.sum() < 100:
        return None
    
    y_train = df_train[train_mask][desc].values
    y_test = df_test[test_mask][desc].values
    X_train_clean = X_train[train_mask]
    X_test_clean = X_test[test_mask]
    
    try:
        # Use appropriate num_workers based on model type
        num_workers = NUM_WORKERS_LINEAR if model_type == 'linear' else NUM_WORKERS_MLP
        
        result = train_probe(
            X_train_clean, y_train,
            X_test_clean, y_test,
            model_type=model_type,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            verbose=False,
            num_workers=num_workers
        )
        
        return {
            'descriptor': desc,
            'r2': result['r2'],
            'mae': result['mae'],
            'n_train': len(y_train),
            'n_test': len(y_test),
            'train_mean': y_train.mean(),
            'train_std': y_train.std(),
            'test_mean': y_test.mean(),
            'test_std': y_test.std()
        }
    except Exception as e:
        print(f"\n⚠️  Failed on {desc}: {e}")
        return None


def probe_descriptor_worker(args):
    """
    Worker function for multiprocessing. This is a wrapper to unpack arguments.
    
    Args:
        args: tuple of (desc, df_train, df_test, X_train, X_test, model_type, device)
    
    Returns:
        dict with results or None if failed
    """
    desc, df_train, df_test, X_train, X_test, model_type, device = args
    
    # Set CUDA device for this process
    if device.startswith('cuda'):
        gpu_id = int(device.split(':')[1])
        torch.cuda.set_device(gpu_id)
    
    return probe_single_descriptor(desc, df_train, df_test, X_train, X_test, model_type, device)


def probe_all_descriptors(devices=None, skip_linear=False, skip_mlp=False):
    """
    Main function to probe all descriptors.
    
    Args:
        devices: List of devices to use (e.g., ['cuda:0', 'cuda:1'] or ['cpu'])
                 If None, automatically detect 2 free GPUs
        skip_linear: Skip linear probing if results exist
        skip_mlp: Skip MLP probing if results exist
    """
    # Auto-detect free GPUs if not specified
    if devices is None:
        if torch.cuda.is_available():
            free_gpu_ids = get_free_gpus(max_gpus=1)  # Changed from 2 to 1
            
            if free_gpu_ids:
                devices = [f'cuda:{i}' for i in free_gpu_ids]
                print(f"✅ Selected 1 free GPU: {devices}\n")
            else:
                # Fallback: use first available GPU
                devices = ['cuda:0']  # Single GPU fallback
                print(f"⚠️  Using first GPU: {devices}")
                print(f"   (Could not detect free GPUs automatically)\n")
        else:
            devices = ['cpu']
            print("🔍 No GPUs detected, using CPU\n")
    
    print("="*80)
    print("COMPREHENSIVE RDKIT DESCRIPTOR PROBING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Devices: {devices} ({len(devices)} device(s))")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("="*80 + "\n")
    
    # Validate devices
    available_devices = []
    for device in devices:
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"⚠️  CUDA not available for {device}, falling back to CPU")
            available_devices.append('cpu')
        else:
            available_devices.append(device)
    
    devices = available_devices
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df_descriptors = pd.read_parquet(DESCRIPTORS_PATH)
    print(f"  Dataset: {len(df):,} spectra")
    print(f"  Descriptors: {len(df_descriptors.columns)-1} total\n")
    
    # Filter valid descriptors
    print("Filtering valid descriptors...")
    descriptor_names = [col for col in df_descriptors.columns if col != 'smiles']
    valid_descriptors = []
    
    for desc in descriptor_names:
        values = df_descriptors[desc]
        nan_pct = values.isna().sum() / len(values)
        
        if nan_pct > 0.1:
            continue
        if np.isinf(values.dropna()).any():
            continue
        if values.dropna().std() == 0:
            continue
        
        valid_descriptors.append(desc)
    
    print(f"  Valid descriptors: {len(valid_descriptors)}/{len(descriptor_names)}\n")
    
    # Merge descriptors with dataset
    print("Merging descriptors with dataset...")
    # Drop any overlapping descriptor columns from the main dataset first
    overlapping_cols = [col for col in valid_descriptors if col in df.columns and col != 'smiles']
    if overlapping_cols:
        print(f"  Dropping {len(overlapping_cols)} overlapping columns: {overlapping_cols[:5]}...")
        df = df.drop(columns=overlapping_cols)
    
    df_with_descriptors = df.merge(
        df_descriptors[['smiles'] + valid_descriptors],
        on='smiles',
        how='left'
    )
    
    # Split by fold
    df_train = df_with_descriptors[df_with_descriptors['fold'] == 'train'].copy()
    df_test = df_with_descriptors[df_with_descriptors['fold'] == 'test'].copy()
    print(f"  Train: {len(df_train):,}")
    print(f"  Test:  {len(df_test):,}\n")
    
    # Extract embeddings
    print("Extracting embeddings...")
    X_train = np.vstack(df_train['ssl_embedding'].values)
    X_test = np.vstack(df_test['ssl_embedding'].values)
    print(f"  Train embeddings: {X_train.shape}")
    print(f"  Test embeddings:  {X_test.shape}\n")
    
    # =============================================================================
    # LINEAR PROBE
    # =============================================================================
    
    results_cache_linear = RESULTS_DIR / 'all_descriptors_probing_results_linear.pkl'
    
    if results_cache_linear.exists() and skip_linear:
        print(f"✓ Skipping Linear probing (results exist)\n")
        with open(results_cache_linear, 'rb') as f:
            all_results_linear = pickle.load(f)
    else:
        print("="*80)
        print("LINEAR PROBE")
        print("="*80)
        print(f"Descriptors: {len(valid_descriptors)}")
        print(f"Devices: {devices}")
        print(f"Estimated time: ~30-60 minutes (with {len(devices)} GPU(s))")
        print("="*80 + "\n")
        
        all_results_linear = []
        
        if len(devices) > 1:
            # Parallel processing across multiple GPUs using processes (not threads)
            print(f"🚀 Using parallel processing with {len(devices)} devices\n")
            
            # Prepare arguments for each descriptor
            worker_args = []
            for i, desc in enumerate(valid_descriptors):
                # Round-robin device assignment
                device = devices[i % len(devices)]
                worker_args.append((desc, df_train, df_test, X_train, X_test, 'linear', device))
            
            # Use ProcessPoolExecutor for true parallelism
            # Set start method to 'spawn' for CUDA compatibility
            mp_context = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp_context) as executor:
                # Submit all jobs
                futures = [executor.submit(probe_descriptor_worker, args) for args in worker_args]
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Linear probing"):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results_linear.append(result)
                    except Exception as e:
                        print(f"\n⚠️  Worker failed: {e}")
        else:
            # Sequential processing (single device)
            for desc in tqdm(valid_descriptors, desc="Linear probing"):
                result = probe_single_descriptor(
                    desc, df_train, df_test, X_train, X_test, 'linear', devices[0]
                )
                if result is not None:
                    all_results_linear.append(result)
        
        # Save Linear results
        with open(results_cache_linear, 'wb') as f:
            pickle.dump(all_results_linear, f)
        
        print(f"\n✅ Linear probing complete!")
        print(f"   Evaluated: {len(all_results_linear)} descriptors")
        print(f"   Saved to: {results_cache_linear.name}\n")
    
    # =============================================================================
    # MLP PROBE
    # =============================================================================
    
    results_cache_mlp = RESULTS_DIR / 'all_descriptors_probing_results_mlp.pkl'
    
    if results_cache_mlp.exists() and skip_mlp:
        print(f"✓ Skipping MLP probing (results exist)\n")
        with open(results_cache_mlp, 'rb') as f:
            all_results_mlp = pickle.load(f)
    else:
        print("="*80)
        print("MLP PROBE")
        print("="*80)
        print(f"Descriptors: {len(valid_descriptors)}")
        print(f"Devices: {devices}")
        print(f"Estimated time: ~60-120 minutes (with {len(devices)} GPU(s))")
        print("="*80 + "\n")
        
        all_results_mlp = []
        
        if len(devices) > 1:
            # Parallel processing across multiple GPUs using processes (not threads)
            print(f"🚀 Using parallel processing with {len(devices)} devices\n")
            
            # Prepare arguments for each descriptor
            worker_args = []
            for i, desc in enumerate(valid_descriptors):
                # Round-robin device assignment
                device = devices[i % len(devices)]
                worker_args.append((desc, df_train, df_test, X_train, X_test, 'mlp', device))
            
            # Use ProcessPoolExecutor for true parallelism
            # Set start method to 'spawn' for CUDA compatibility
            mp_context = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp_context) as executor:
                # Submit all jobs
                futures = [executor.submit(probe_descriptor_worker, args) for args in worker_args]
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="MLP probing"):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results_mlp.append(result)
                    except Exception as e:
                        print(f"\n⚠️  Worker failed: {e}")
        else:
            # Sequential processing (single device)
            for desc in tqdm(valid_descriptors, desc="MLP probing"):
                result = probe_single_descriptor(
                    desc, df_train, df_test, X_train, X_test, 'mlp', devices[0]
                )
                if result is not None:
                    all_results_mlp.append(result)
        
        # Save MLP results
        with open(results_cache_mlp, 'wb') as f:
            pickle.dump(all_results_mlp, f)
        
        print(f"\n✅ MLP probing complete!")
        print(f"   Evaluated: {len(all_results_mlp)} descriptors")
        print(f"   Saved to: {results_cache_mlp.name}\n")
    
    # =============================================================================
    # Summary
    # =============================================================================
    
    print("="*80)
    print("PROBING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  - {results_cache_linear}")
    print(f"  - {results_cache_mlp}")
    print("\nNext steps:")
    print("  1. Download these .pkl files to your local machine")
    print("  2. Run the analysis cells in the Jupyter notebook")
    print("="*80)
    
    # Save summary
    summary_path = RESULTS_DIR / 'cluster_run_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE RDKIT DESCRIPTOR PROBING - CLUSTER RUN SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Devices: {devices} ({len(devices)} device(s))\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Dropout: {DROPOUT}\n\n")
        f.write(f"Dataset size: {len(df):,} spectra\n")
        f.write(f"Train size: {len(df_train):,}\n")
        f.write(f"Test size: {len(df_test):,}\n\n")
        f.write(f"Total descriptors: {len(descriptor_names)}\n")
        f.write(f"Valid descriptors: {len(valid_descriptors)}\n")
        f.write(f"Linear results: {len(all_results_linear)}\n")
        f.write(f"MLP results: {len(all_results_mlp)}\n\n")
        f.write("Output files:\n")
        f.write(f"  - {results_cache_linear.name}\n")
        f.write(f"  - {results_cache_mlp.name}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Summary saved to: {summary_path}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe ALL RDKit descriptors with Linear and MLP models. "
                    "Automatically detects and uses 2 free GPUs."
    )
    parser.add_argument(
        '--skip-linear',
        action='store_true',
        help='Skip linear probing if results already exist'
    )
    parser.add_argument(
        '--skip-mlp',
        action='store_true',
        help='Skip MLP probing if results already exist'
    )
    
    args = parser.parse_args()
    
    try:
        probe_all_descriptors(
            devices=None,  # Auto-detect GPUs
            skip_linear=args.skip_linear,
            skip_mlp=args.skip_mlp
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
