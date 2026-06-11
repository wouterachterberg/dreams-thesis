"""
Add DreaMS embeddings to existing TSV file with scaffold splits.

This script takes the existing MassSpecGym_scaffold_splits.tsv file and adds
embeddings from the .npy files, maintaining the same order.

Usage:
    python add_embeddings_to_tsv.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def add_embeddings_to_tsv(
    tsv_path: Path,
    embeddings_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Add DreaMS embeddings to existing TSV file with scaffold splits.
    
    The TSV file and embeddings are both ordered by the original MassSpecGym identifiers,
    so we can match them by position.
    
    Args:
        tsv_path: Path to MassSpecGym_scaffold_splits.tsv
        embeddings_dir: Directory containing val_chunk_*.npy embedding files
        output_path: Where to save the enhanced dataset
        
    Returns:
        DataFrame with embeddings added
    """
    print("Loading existing TSV file with scaffold splits...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"  Loaded {len(df):,} samples")
    print(f"  Columns: {len(df.columns)} columns")
    
    # Load all embeddings in order
    embedding_files = sorted(embeddings_dir.glob('val_chunk_*.npy'))
    print(f"\nLoading embeddings from {len(embedding_files)} chunks...")
    
    all_embeddings = []
    for emb_file in tqdm(embedding_files, desc="Loading embeddings"):
        embeddings = np.load(emb_file)
        all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"  Total embeddings loaded: {len(all_embeddings):,}")
    print(f"  Embedding dimension: {all_embeddings.shape[1]}")
    
    # Verify same number of samples
    assert len(df) == len(all_embeddings), \
        f"Mismatch: TSV has {len(df):,} rows, embeddings have {len(all_embeddings):,}"
    
    # Add embeddings as a new column
    print("\nAdding embeddings to dataframe...")
    df['embedding'] = list(all_embeddings)
    
    print(f"\n✅ Enhanced dataset created:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Embedding dimension: {all_embeddings.shape[1]}")
    
    # Show distribution across folds
    print(f"\n  Distribution by fold:")
    for fold, count in df['fold'].value_counts().sort_index().items():
        print(f"    {fold}: {count:,} samples")
    
    # Save to parquet (better for storing arrays than TSV)
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("✅ Saved!")
    
    # Also show what targets are available
    print(f"\n  Available targets for probing:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"    {', '.join(numeric_cols)}")
    
    return df


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent / 'data'
    tsv_path = base_dir / 'processed' / 'MassSpecGym_scaffold_splits.tsv'
    embeddings_dir = base_dir / 'processed' / 'massspecgym_complete' / 'val_embs'
    output_path = base_dir / 'processed' / 'MassSpecGym_with_embeddings.parquet'
    
    print("=" * 80)
    print("ADDING EMBEDDINGS TO EXISTING SCAFFOLD SPLITS TSV")
    print("=" * 80)
    
    df = add_embeddings_to_tsv(
        tsv_path=tsv_path,
        embeddings_dir=embeddings_dir,
        output_path=output_path
    )
    
    print("\n" + "=" * 80)
    print("DATASET READY FOR PROBING!")
    print("=" * 80)
    print(f"\nYou now have a single file: {output_path}")
    print(f"\nThis file contains:")
    print(f"  - 'embedding': DreaMS embedding vectors (1024-dim)")
    print(f"  - 'fold': train/val/test splits (scaffold-based)")
    print(f"  - 'mol_weight', 'logp', 'tpsa': Molecular properties for probing")
    print(f"  - 'scaffold_id': For tracking scaffolds")
    print(f"  - All other molecular features and metadata")
    print(f"\nExample usage:")
    print(f"  df = pd.read_parquet('{output_path}')")
    print(f"  train_df = df[df['fold'] == 'train']")
    print(f"  X_train = np.vstack(train_df['embedding'].values)")
    print(f"  y_train = train_df['mol_weight'].values")
