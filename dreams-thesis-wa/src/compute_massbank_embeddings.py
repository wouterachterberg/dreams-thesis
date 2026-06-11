"""
Generate SSL embeddings for MassBank EU external test set.

This script uses the existing generate_ssl_embeddings.py infrastructure
to compute SSL embeddings for the MassBank external validation set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the existing embedding generation function
from generate_ssl_embeddings import generate_ssl_embeddings


def prepare_massbank_for_embedding_generation(input_parquet: str, output_parquet: str, top_n_peaks: int = 100):
    """
    Prepare MassBank data to match the format expected by generate_ssl_embeddings.
    
    The generate_ssl_embeddings function expects:
    - 'mzs': comma-separated string of m/z values
    - 'intensities': comma-separated string of intensity values
    
    MassBank data currently has:
    - 'mzs': numpy array
    - 'intensities': numpy array
    
    We also trim to top N peaks (by intensity) to match MassSpecGym processing.
    """
    print("Preparing MassBank data format...")
    df = pd.read_parquet(input_parquet)
    
    print(f"Loaded {len(df)} compounds from MassBank")
    print(f"Trimming to top {top_n_peaks} peaks by intensity...")
    
    def trim_spectrum(row):
        """Keep only top N peaks by intensity"""
        mzs = row['mzs']
        intensities = row['intensities']
        
        # Sort by intensity (descending)
        indices = np.argsort(intensities)[::-1][:top_n_peaks]
        # Re-sort by m/z for proper spectrum format
        indices = indices[np.argsort(mzs[indices])]
        
        return mzs[indices], intensities[indices]
    
    # Trim spectra
    trimmed = df.apply(trim_spectrum, axis=1, result_type='expand')
    df['mzs'] = trimmed[0]
    df['intensities'] = trimmed[1]
    
    # Update n_peaks
    df['n_peaks'] = df['mzs'].apply(len)
    
    print(f"After trimming: mean peaks = {df['n_peaks'].mean():.1f}, max = {df['n_peaks'].max()}")
    
    # Convert arrays to comma-separated strings
    df['mzs'] = df['mzs'].apply(lambda x: ','.join(map(str, x)))
    df['intensities'] = df['intensities'].apply(lambda x: ','.join(map(str, x)))
    
    # Save temporary file
    df.to_parquet(output_parquet, index=False)
    print(f"Saved formatted data to {output_parquet}")
    
    return df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    external_dir = base_dir / "data" / "external" / "massbank_eu"
    
    input_file = external_dir / "massbank_eu_external_test_no_embeddings.parquet"
    temp_file = external_dir / "massbank_eu_formatted.parquet"
    output_file = external_dir / "massbank_eu_external_test.parquet"
    
    print("="*80)
    print("GENERATING SSL EMBEDDINGS FOR MASSBANK EU")
    print("="*80)
    print(f"Input:  {input_file.name}")
    print(f"Output: {output_file.name}")
    print("="*80 + "\n")
    
    # Step 1: Prepare data format
    df = prepare_massbank_for_embedding_generation(
        input_parquet=str(input_file),
        output_parquet=str(temp_file)
    )
    
    print(f"\n{'='*80}")
    print("GENERATING EMBEDDINGS WITH FROZEN SSL MODEL")
    print("="*80 + "\n")
    
    # Step 2: Generate embeddings using existing function
    df_with_embeddings = generate_ssl_embeddings(
        input_parquet=str(temp_file),
        output_parquet=str(output_file),
        batch_size=32,  # Same as MassSpecGym processing
        device=None  # Auto-detect
    )
    
    # Clean up temp file
    temp_file.unlink()
    
    print(f"\n{'='*80}")
    print("MASSBANK EXTERNAL TEST SET READY!")
    print("="*80)
    print(f"✅ Generated SSL embeddings for {len(df_with_embeddings)} MassBank compounds")
    print(f"   Embedding dimension: {df_with_embeddings['ssl_embedding'].iloc[0].shape[0]}")
    print(f"   Saved to: {output_file}")
    print(f"\n📊 Dataset statistics:")
    print(f"   Unique InChIKeys: {df_with_embeddings['inchikey'].nunique()}")
    print(f"   Mean peaks per spectrum: {df_with_embeddings['n_peaks'].mean():.1f}")
    print(f"\n📝 Next step:")
    print(f"   Run external validation notebook to evaluate probes on this dataset")
    print("="*80)
