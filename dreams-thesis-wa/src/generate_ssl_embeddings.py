"""
Generate pure SSL embeddings (without contrastive fine-tuning).

This script:
1. Loads ssl_model.ckpt directly (not through embedding_model.ckpt)
2. Extracts embeddings from the SSL Transformer backbone only
3. Saves embeddings to ssl_embs/ folder for comparison against contrastive embeddings
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import dreams
sys.path.append(str(Path(__file__).parent.parent.parent))

from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS


def generate_ssl_embeddings(
    input_parquet: str,
    output_parquet: str,
    batch_size: int = 32,  # Reduced from 128 for faster processing
    device: str = None
):
    """
    Generate pure SSL embeddings without contrastive fine-tuning.
    
    Args:
        input_parquet: Path to input parquet with spectra
        output_parquet: Path to save output with SSL embeddings
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
    """
    
    if device is None:
        # Check for GPU: CUDA (NVIDIA) or MPS (Apple Silicon)
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load the pure SSL model (NOT the contrastive fine-tuned version)
    print("\nLoading pure SSL model (ssl_model.ckpt)...")
    
    # Get path to ssl_model.ckpt
    from dreams.definitions import PRETRAINED
    ssl_ckpt_path = PRETRAINED / 'ssl_model.ckpt'
    
    model = PreTrainedModel.from_ckpt(
        ckpt_path=ssl_ckpt_path,
        ckpt_cls=DreaMS,  # Load as DreaMS model directly
        n_highest_peaks=100,
        remove_unused_backbone_parameters=True
    )
    model.model.eval()
    
    # Move model to device
    model.model = model.model.to(device)
    
    print(f"Model loaded: {type(model.model)}")
    print(f"Model has {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    # Load data
    print(f"\nLoading data from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df):,} samples")
    
    # Generate embeddings
    print(f"\nGenerating SSL embeddings (batch_size={batch_size})...")
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare spectra from mzs and intensities columns
            specs = []
            for idx in range(len(batch)):
                row = batch.iloc[idx]
                # Parse comma-separated strings to arrays
                mz = torch.tensor([float(x) for x in row['mzs'].split(',')], dtype=torch.float32)
                intensity = torch.tensor([float(x) for x in row['intensities'].split(',')], dtype=torch.float32)
                spec = torch.stack([mz, intensity], dim=1)
                specs.append(spec)
            
            # Pad to same length
            max_len = max(s.shape[0] for s in specs)
            padded_specs = []
            for spec in specs:
                if spec.shape[0] < max_len:
                    padding = torch.zeros(max_len - spec.shape[0], 2)
                    spec = torch.cat([spec, padding], dim=0)
                padded_specs.append(spec)
            
            # Stack into batch
            batch_specs = torch.stack(padded_specs).to(device)
            
            # Get embeddings from SSL model
            # DreaMS forward returns (batch_size, seq_len, d_model)
            # We extract the precursor token embeddings [:, 0, :]
            embs = model.model(batch_specs)
            
            # Extract precursor token (first token)
            precursor_embs = embs[:, 0, :].cpu().numpy()
            
            all_embeddings.append(precursor_embs)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"\nGenerated embeddings shape: {all_embeddings.shape}")
    print(f"Embedding statistics: mean={all_embeddings.mean():.4f}, std={all_embeddings.std():.4f}")
    
    # Add embeddings to dataframe
    df['ssl_embedding'] = list(all_embeddings)
    
    # Save to parquet
    print(f"\nSaving to {output_parquet}...")
    df.to_parquet(output_parquet, index=False)
    print("Done!")
    
    return df


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed"
    output_dir = data_dir / "ssl_embs"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = data_dir / "MassSpecGym_with_embeddings.parquet"
    output_file = output_dir / "MassSpecGym_with_SSL_embeddings.parquet"
    
    print(f"{'='*60}")
    print("Generating Pure SSL Embeddings")
    print(f"{'='*60}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Generate embeddings
    df = generate_ssl_embeddings(
        input_parquet=str(input_file),
        output_parquet=str(output_file),
        batch_size=32,  # Reduced from 128 for faster processing
        device=None  # Auto-detect: will use MPS (Apple GPU) if available
    )
    
    print(f"\n{'='*60}")
    print("SSL Embeddings Generated Successfully!")
    print(f"{'='*60}")
    print(f"Total samples: {len(df):,}")
    print(f"Embedding dimension: {df['ssl_embedding'].iloc[0].shape[0]}")
    print(f"\nThese are PURE SSL embeddings (no contrastive fine-tuning)")
    print("Compare with 'embedding' column for contrastive embeddings")
    print(f"{'='*60}")
