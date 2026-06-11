ssh# Running Descriptor Probing on GPU Cluster

This guide explains how to run the computationally intensive probing on a GPU cluster and analyze results locally.

## 📁 Files

1. **`src/probe_all_descriptors_cluster.py`** - Standalone Python script for cluster
2. **`notebooks/exploratory/probe_all_rdkit_descriptors.ipynb`** - Jupyter notebook for local analysis

## 🚀 Workflow

### Step 1: Prepare Data on Cluster

Upload these files to your cluster:
```bash
# From your local machine
scp -r data/processed/massspecgym_complete your_cluster:/path/to/project/data/processed/
```

Required files:
- `MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet`
- `all_rdkit_descriptors.parquet`

### Step 2: Upload Script

```bash
# Upload the probing script to the cluster
scp src/probe_all_descriptors_cluster.py your_cluster:/path/to/project/src/
```

### Step 3: Run on Cluster

```bash
# SSH to cluster
ssh your_cluster

# Navigate to project directory
cd /path/to/project/src

# Run directly (will use GPU by default)
python probe_all_descriptors_cluster.py --device cuda:0

# Or run on CPU if needed
python probe_all_descriptors_cluster.py --device cpu

# Run in background with nohup
nohup python probe_all_descriptors_cluster.py --device cuda:0 > ../logs/probing.log 2>&1 &

# Monitor progress
tail -f ../logs/probing.log
```

### Step 4: Monitor Progress

The script will print progress updates:
```
================================================================================
LINEAR PROBE
================================================================================
Descriptors: 201
Estimated time: ~30-60 minutes
================================================================================

Linear probing: 100%|██████████| 201/201 [45:23<00:00, 13.51s/it]

✅ Linear probing complete!
   Evaluated: 201 descriptors
   Saved to: all_descriptors_probing_results_linear.pkl
```

### Step 5: Download Results

Once the job completes, download the results:
```bash
# From your local machine
scp your_cluster:/path/to/results/*.pkl results/
scp your_cluster:/path/to/results/cluster_run_summary.txt results/
```

Files to download:
- `all_descriptors_probing_results_linear.pkl` (~2-5 MB)
- `all_descriptors_probing_results_mlp.pkl` (~10-20 MB)
- `cluster_run_summary.txt` (metadata)

### Step 6: Analyze Locally

Open `probe_all_rdkit_descriptors.ipynb` and run **only the analysis cells**:
- Skip cells 1-16 (data loading and probing)
- Start from cell 17 (Analyze Results)
- The notebook will load the `.pkl` files from `results/`

## ⚙️ Advanced Options

### Run on CPU
```bash
python probe_all_descriptors_cluster.py --device cpu
```

### Skip completed probes
If a probe crashes and you want to skip already-completed parts:
```bash
python probe_all_descriptors_cluster.py --skip-linear  # Only run MLP
python probe_all_descriptors_cluster.py --skip-mlp     # Only run Linear
```

### Test run (single descriptor)
Edit the script and add this before the probing loops:
```python
valid_descriptors = valid_descriptors[:1]  # Test with just 1 descriptor
```

## 🔧 Troubleshooting

### Out of Memory
Reduce batch size in `probe_all_descriptors_cluster.py`:
```python
BATCH_SIZE = 128  # or 64
```

### CUDA not available
Check if GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi  # Should show GPU info
```

### Import errors
Ensure PyTorch, scikit-learn, pandas are installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas pyarrow tqdm
```

## ⏱️ Expected Runtime

| Phase | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| Linear probing (201 descriptors) | ~30-45 min | ~45-90 min |
| MLP probing (201 descriptors) | ~60-90 min | ~2-3 hours |
| **Total** | **~90-135 min** | **~3-4.5 hours** |

GPU speeds up MLP training significantly, but Linear probing is already fast on CPU.

## 📊 Output Files

### PKL Files
Binary pickle files containing:
```python
[
    {
        'descriptor': 'MolWt',
        'r2': 0.823,
        'mae': 45.2,
        'n_train': 159271,
        'n_test': 45185,
        'train_mean': 357.4,
        'train_std': 178.9,
        'test_mean': 358.1,
        'test_std': 179.3
    },
    # ... 200+ more descriptors
]
```

### Summary File
Text file with run metadata:
- Device used
- Hyperparameters
- Dataset sizes
- Completion time

## 🎯 Benefits of This Approach

✅ **GPU acceleration**: MLP training is 2-3× faster  
✅ **No notebook state issues**: Clean Python script  
✅ **Easy to rerun**: Cached results, can skip completed parts  
✅ **Portable**: Works on any GPU cluster with Python  
✅ **Local analysis**: Keep visualization/analysis on your laptop  

## 📝 Quick Start Summary

```bash
# On local machine - upload script
scp src/probe_all_descriptors_cluster.py cluster:/path/to/project/src/

# On cluster - run probing
ssh cluster
cd /path/to/project/src
python probe_all_descriptors_cluster.py --device cuda:0

# On local machine - download results
scp cluster:/path/to/project/results/*.pkl results/
```

Then open `probe_all_rdkit_descriptors.ipynb` and run cells 17-30 (analysis and visualization).
