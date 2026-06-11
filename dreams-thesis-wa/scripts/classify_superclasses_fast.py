"""
Fast parallel ClassyFire/NPClassifier classification using ThreadPoolExecutor.

Estimated time for 31,602 unique SMILES:
- 1 worker (current): ~9 hours
- 20 workers (this script): ~30-45 minutes
- 50 workers (aggressive): ~15-20 minutes

Usage:
    python classify_superclasses_fast.py
"""

import pandas as pd
import urllib.request
import urllib.parse
import json
import time
import h5py
from pathlib import Path
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
SPLITS_DIR = DATA_DIR / "MassSpecGym_splits"
FULL_HDF5 = SPLITS_DIR / "full.hdf5"
CACHE_FILE = SPLITS_DIR / "smiles_superclass_cache.pkl"
OUTPUT_DIR = SPLITS_DIR

# Parallel settings - adjust based on API tolerance
MAX_WORKERS = 30  # Number of parallel requests

# Thread-safe cache lock
cache_lock = threading.Lock()


def load_cache() -> dict:
    """Load cached classification results."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
            print(f"Loaded {len(cache)} cached classifications")
            return cache
    return {}


def save_cache(cache: dict):
    """Save classification cache."""
    with cache_lock:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)


def classify_smiles(smiles: str, retry_count: int = 3) -> tuple:
    """Classify a single SMILES. Returns (smiles, superclass)."""
    url = f"https://npclassifier.ucsd.edu/classify?smiles={urllib.parse.quote(smiles)}"
    
    for attempt in range(retry_count):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                result = json.loads(response.read().decode())
                superclass_list = result.get('superclass_results', [])
                if superclass_list and len(superclass_list) > 0:
                    return smiles, superclass_list[0]
                return smiles, "Unclassified"
        except Exception as e:
            if attempt == retry_count - 1:
                return smiles, "Error"
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    return smiles, "Error"


def load_smiles_and_folds() -> pd.DataFrame:
    """Load unique SMILES and their fold assignments from full.hdf5."""
    print("Loading dataset from full.hdf5...")
    
    with h5py.File(FULL_HDF5, 'r') as hf:
        smiles = hf['smiles'][:]
        folds = hf['fold'][:]
        
        if isinstance(smiles[0], bytes):
            smiles = [s.decode() for s in smiles]
        if isinstance(folds[0], bytes):
            folds = [f.decode() for f in folds]
    
    df = pd.DataFrame({'smiles': smiles, 'fold': folds})
    print(f"Loaded {len(df):,} spectra")
    
    # Get unique SMILES with their fold
    unique_df = df.groupby('smiles').agg({'fold': 'first'}).reset_index()
    print(f"Found {len(unique_df):,} unique SMILES")
    
    return unique_df


def classify_all_parallel(smiles_list: list, cache: dict) -> dict:
    """Classify all SMILES using parallel threads."""
    results = {}
    to_classify = []
    
    # Check cache first
    for smiles in smiles_list:
        if smiles in cache:
            results[smiles] = cache[smiles]
        else:
            to_classify.append(smiles)
    
    print(f"Found {len(results):,} in cache, need to classify {len(to_classify):,} new SMILES")
    
    if not to_classify:
        return results
    
    # Estimate time
    estimated_minutes = len(to_classify) / MAX_WORKERS * 0.8 / 60
    print(f"Estimated time with {MAX_WORKERS} workers: ~{estimated_minutes:.0f} minutes")
    
    # Process with thread pool
    completed_count = 0
    save_interval = 500
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_smiles = {executor.submit(classify_smiles, s): s for s in to_classify}
        
        # Process as they complete
        with tqdm(total=len(to_classify), desc="Classifying") as pbar:
            for future in as_completed(future_to_smiles):
                smiles, superclass = future.result()
                results[smiles] = superclass
                
                with cache_lock:
                    cache[smiles] = superclass
                
                completed_count += 1
                pbar.update(1)
                
                # Save cache periodically
                if completed_count % save_interval == 0:
                    save_cache(cache)
                    tqdm.write(f" [Saved cache at {completed_count:,}/{len(to_classify):,}]")
    
    # Final save
    save_cache(cache)
    print(f"\nClassification complete. Total: {len(results):,}")
    
    return results


def create_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Create summary tables for thesis."""
    
    # 1. Cross-tabulation by fold
    crosstab = pd.crosstab(
        df['superclass'], 
        df['fold'], 
        margins=True, 
        margins_name='Total'
    ).sort_values('Total', ascending=False)
    
    crosstab.to_csv(output_dir / "superclass_by_fold.csv")
    print(f"\nSaved: superclass_by_fold.csv")
    
    # 2. Summary statistics
    print("\n" + "="*60)
    print("SUPERCLASS DISTRIBUTION SUMMARY")
    print("="*60)
    print(f"\nTop 15 superclasses:")
    print(crosstab.head(15).to_string())
    
    # 3. LaTeX table
    latex = crosstab.head(15).to_latex(
        caption="Distribution of NPClassifier superclasses across train and validation folds in MassSpecGym dataset.",
        label="tab:superclass_distribution",
        column_format='l' + 'r' * len(crosstab.columns)
    )
    
    with open(output_dir / "superclass_by_fold.tex", 'w') as f:
        f.write(latex)
    print(f"Saved: superclass_by_fold.tex")
    
    # 4. Save full mapping
    df.to_csv(output_dir / "smiles_with_superclass.csv", index=False)
    print(f"Saved: smiles_with_superclass.csv")
    
    return crosstab


def main():
    start_time = time.time()
    
    # Load data
    unique_df = load_smiles_and_folds()
    
    # Load cache
    cache = load_cache()
    
    # Classify all SMILES in parallel
    smiles_list = unique_df['smiles'].tolist()
    classification_results = classify_all_parallel(smiles_list, cache)
    
    # Add superclass to dataframe
    unique_df['superclass'] = unique_df['smiles'].map(classification_results)
    
    # Create summary tables
    crosstab = create_summary_tables(unique_df, OUTPUT_DIR)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE! Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"{'='*60}")
    
    return unique_df, crosstab


if __name__ == "__main__":
    result_df, crosstab = main()
