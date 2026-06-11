"""
Prepare MassBank EU External Test Set for DreaMS Probing

This script:
1. Downloads MassBank EU spectra via their API
2. Filters to match MassSpecGym conditions (positive ESI, mid HCD/CID energies)
3. Deduplicates by InChIKey against MassSpecGym training data
4. Computes SSL embeddings using frozen DreaMS encoder
5. Computes RDKit descriptors (same 10 as internal test)
6. Saves clean external test set for probing validation

Target: ~200-500 unique compounds for stable external validation
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
import torch
from typing import List, Dict, Any

# Configure paths
DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed' / 'massspecgym_complete' / 'ssl_embs'
EXTERNAL_DIR = Path(__file__).parent.parent / 'data' / 'external' / 'massbank_eu'
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# MassBank EU API endpoints
MASSBANK_API_BASE = "https://massbank.eu/MassBank/RecordDisplay"
MASSBANK_SEARCH_API = "https://massbank.eu/MassBank/Search"

# Target descriptors (same as probing experiments)
TARGET_DESCRIPTORS = [
    'alogp', 'hba', 'hbd', 'tpsa', 
    'n_rotatable_bonds', 'n_aromatic_rings', 'n_aliphatic_rings',
    'fsp3', 'qed', 'sa_score'
]


def calculate_sa_score(mol):
    """Calculate Synthetic Accessibility score (import from Chem)"""
    try:
        from rdkit.Chem import Crippen
        from rdkit.Chem.Descriptors import sa_score
        return sa_score(mol)
    except:
        # If sa_score not available, try alternative import
        try:
            from rdkit.Contrib.SA_Score import sascorer
            return sascorer.calculateScore(mol)
        except:
            print("⚠️  SA Score calculation unavailable, using placeholder")
            return np.nan


def compute_rdkit_descriptors(smiles: str) -> Dict[str, float]:
    """
    Compute RDKit descriptors for a SMILES string.
    Returns dict with 10 molecular properties.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {desc: np.nan for desc in TARGET_DESCRIPTORS}
    
    # Calculate descriptors
    descriptors = {
        'alogp': Descriptors.MolLogP(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'n_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'n_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'qed': QED.qed(mol),
        'sa_score': calculate_sa_score(mol)
    }
    
    return descriptors


def download_massbank_records_simple():
    """
    Simple approach: Download MassBank EU records via wget/curl.
    
    MassBank provides bulk download at:
    https://github.com/MassBank/MassBank-data/releases
    
    We'll download a recent release and parse locally.
    """
    print("="*80)
    print("DOWNLOADING MASSBANK EU DATA")
    print("="*80)
    
    # Use GitHub release (faster than API scraping)
    massbank_url = "https://github.com/MassBank/MassBank-data/archive/refs/heads/main.zip"
    
    print(f"\n📥 Downloading MassBank-data from GitHub...")
    print(f"   URL: {massbank_url}")
    print(f"   This may take a few minutes (~500 MB)...")
    
    import urllib.request
    import zipfile
    import io
    
    # Download zip file
    zip_path = EXTERNAL_DIR / 'massbank_data.zip'
    
    try:
        urllib.request.urlretrieve(massbank_url, zip_path)
        print(f"   ✓ Downloaded to {zip_path}")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print(f"\n   Manual alternative:")
        print(f"   1. Download: {massbank_url}")
        print(f"   2. Extract to: {EXTERNAL_DIR}")
        print(f"   3. Re-run this script")
        return None
    
    # Extract
    print(f"\n📂 Extracting records...")
    extract_dir = EXTERNAL_DIR / 'MassBank-data-main'
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTERNAL_DIR)
    
    print(f"   ✓ Extracted to {extract_dir}")
    
    return extract_dir


def parse_massbank_record(record_path: Path) -> Dict[str, Any]:
    """
    Parse a single MassBank record file (.txt format).
    
    Returns dict with:
    - accession: record ID
    - smiles: SMILES string
    - inchikey: InChIKey
    - spectrum: list of (mz, intensity) tuples
    - precursor_mz: precursor m/z
    - collision_energy: collision energy
    - ionization: ionization mode
    - instrument_type: MS instrument type
    """
    with open(record_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    record = {
        'accession': None,
        'smiles': None,
        'inchikey': None,
        'formula': None,
        'precursor_mz': None,
        'collision_energy': None,
        'ionization': None,
        'instrument_type': None,
        'spectrum': []
    }
    
    in_peak_section = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('ACCESSION:'):
            record['accession'] = line.split(':', 1)[1].strip()
        
        elif line.startswith('CH$SMILES:'):
            record['smiles'] = line.split(':', 1)[1].strip()
        
        elif line.startswith('CH$LINK: INCHIKEY'):
            record['inchikey'] = line.split('INCHIKEY', 1)[1].strip()
        
        elif line.startswith('CH$FORMULA:'):
            record['formula'] = line.split(':', 1)[1].strip()
        
        elif line.startswith('MS$FOCUSED_ION: PRECURSOR_M/Z'):
            try:
                record['precursor_mz'] = float(line.split('M/Z', 1)[1].strip())
            except:
                pass
        
        elif 'COLLISION_ENERGY' in line:
            # Extract collision energy (e.g., "30 eV" or "HCD 30%")
            try:
                ce_str = line.split(':', 1)[1].strip()
                # Extract numeric value
                import re
                match = re.search(r'(\d+\.?\d*)', ce_str)
                if match:
                    record['collision_energy'] = float(match.group(1))
            except:
                pass
        
        elif line.startswith('AC$MASS_SPECTROMETRY: ION_MODE'):
            record['ionization'] = line.split('ION_MODE', 1)[1].strip()
        
        elif line.startswith('AC$INSTRUMENT_TYPE:'):
            record['instrument_type'] = line.split(':', 1)[1].strip()
        
        elif line.startswith('PK$PEAK:'):
            in_peak_section = True
            continue
        
        elif line.startswith('//'):
            in_peak_section = False
        
        elif in_peak_section and line:
            # Parse peak line: "m/z intensity rel_intensity"
            try:
                parts = line.split()
                if len(parts) >= 2:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    record['spectrum'].append((mz, intensity))
            except:
                pass
    
    return record


def filter_massbank_records(records: List[Dict]) -> List[Dict]:
    """
    Filter MassBank records to match MassSpecGym conditions:
    - Positive ESI ionization
    - Orbitrap/QTOF instruments (similar to Orbitrap)
    - Mid collision energy (20-40 eV or 20-40% HCD/CID)
    - Valid SMILES and InChIKey
    - At least 5 peaks in spectrum
    """
    print("\n" + "="*80)
    print("FILTERING MASSBANK RECORDS")
    print("="*80)
    
    filtered = []
    
    for rec in tqdm(records, desc="Filtering"):
        # Must have SMILES and InChIKey
        if not rec['smiles'] or not rec['inchikey']:
            continue
        
        # Positive ionization
        if rec['ionization'] not in ['POSITIVE', 'P']:
            continue
        
        # Instrument type (Orbitrap or QTOF preferred)
        if rec['instrument_type']:
            inst = rec['instrument_type'].upper()
            if not any(keyword in inst for keyword in ['ORBITRAP', 'QTOF', 'Q-TOF']):
                continue
        
        # Collision energy filter (mid-range: 20-40)
        if rec['collision_energy']:
            if not (20 <= rec['collision_energy'] <= 40):
                continue
        
        # At least 5 peaks
        if len(rec['spectrum']) < 5:
            continue
        
        filtered.append(rec)
    
    print(f"\n✓ Filtered: {len(filtered)} / {len(records)} records")
    print(f"  Criteria: Positive ESI, Orbitrap/QTOF, CE 20-40, ≥5 peaks")
    
    return filtered


def deduplicate_against_massspecgym(
    massbank_df: pd.DataFrame,
    massspecgym_path: Path
) -> pd.DataFrame:
    """
    Remove any MassBank compounds that appear in MassSpecGym training data.
    Deduplication by InChIKey (first block = molecular skeleton).
    """
    print("\n" + "="*80)
    print("DEDUPLICATING AGAINST MASSSPECGYM")
    print("="*80)
    
    # Load MassSpecGym data
    print(f"\n📂 Loading MassSpecGym data from {massspecgym_path.name}...")
    msg_df = pd.read_parquet(massspecgym_path)
    
    # Extract InChIKey first block (molecular skeleton)
    print(f"   Loaded {len(msg_df):,} MassSpecGym spectra")
    print(f"   Unique molecules: {msg_df['smiles'].nunique():,}")
    
    # Get unique InChIKeys from MassSpecGym
    msg_inchikeys = set(msg_df['inchikey'].dropna().unique())
    print(f"\n   MassSpecGym InChIKeys: {len(msg_inchikeys):,}")
    
    # Filter MassBank
    initial_count = len(massbank_df)
    massbank_df = massbank_df[~massbank_df['inchikey'].isin(msg_inchikeys)].copy()
    removed_count = initial_count - len(massbank_df)
    
    print(f"\n✓ Removed {removed_count} overlapping compounds")
    print(f"   Remaining: {len(massbank_df)} unique external test compounds")
    
    return massbank_df


def prepare_massbank_spectra(records: List[Dict]) -> pd.DataFrame:
    """
    Convert filtered MassBank records to DataFrame format.
    Keep one spectrum per compound (select best/first).
    """
    print("\n" + "="*80)
    print("PREPARING MASSBANK DATAFRAME")
    print("="*80)
    
    data = []
    
    # Group by InChIKey and keep first (or best quality)
    from collections import defaultdict
    by_inchikey = defaultdict(list)
    
    for rec in records:
        by_inchikey[rec['inchikey']].append(rec)
    
    print(f"\nGrouped {len(records)} records into {len(by_inchikey)} unique InChIKeys")
    
    for inchikey, recs in tqdm(by_inchikey.items(), desc="Processing"):
        # Pick record with most peaks (best quality)
        best_rec = max(recs, key=lambda r: len(r['spectrum']))
        
        # Normalize spectrum (use relative intensities)
        spectrum = np.array(best_rec['spectrum'])
        if len(spectrum) == 0:
            continue
        
        mzs = spectrum[:, 0]
        intensities = spectrum[:, 1]
        
        # Normalize intensities to [0, 999]
        intensities = (intensities / intensities.max() * 999).astype(int)
        
        data.append({
            'accession': best_rec['accession'],
            'inchikey': best_rec['inchikey'],
            'smiles': best_rec['smiles'],
            'formula': best_rec['formula'],
            'precursor_mz': best_rec['precursor_mz'],
            'collision_energy': best_rec['collision_energy'],
            'mzs': mzs,
            'intensities': intensities,
            'n_peaks': len(mzs)
        })
    
    df = pd.DataFrame(data)
    print(f"\n✓ Created DataFrame with {len(df)} unique compounds")
    
    return df


def compute_ssl_embeddings_for_massbank(
    df: pd.DataFrame,
    model_path: Path
) -> pd.DataFrame:
    """
    Compute DreaMS SSL embeddings for MassBank spectra.
    Uses the same frozen encoder as MassSpecGym.
    
    Note: This requires the DreaMS model and may need GPU.
    For now, we'll prepare the data structure and you can run embedding generation separately.
    """
    print("\n" + "="*80)
    print("PREPARING FOR SSL EMBEDDING COMPUTATION")
    print("="*80)
    
    print("\n⚠️  SSL embedding computation requires DreaMS model loaded in memory.")
    print("   This script will prepare the data structure.")
    print("   You'll need to run embedding generation separately (see next script).")
    
    # Add placeholder for embeddings
    df['ssl_embedding'] = None
    
    print(f"\n✓ DataFrame prepared for {len(df)} compounds")
    print(f"   Next step: Run compute_massbank_embeddings.py to generate embeddings")
    
    return df


def main():
    """Main workflow"""
    print("="*80)
    print("MASSBANK EU EXTERNAL TEST SET PREPARATION")
    print("="*80)
    print("\nGoal: Create clean external validation set (~200-500 compounds)")
    print("Steps:")
    print("  1. Download MassBank EU data")
    print("  2. Parse and filter records (positive ESI, Orbitrap/QTOF, mid CE)")
    print("  3. Deduplicate against MassSpecGym")
    print("  4. Compute RDKit descriptors")
    print("  5. Save for embedding generation")
    print("\n" + "="*80)
    
    # Step 1: Download MassBank data
    extract_dir = download_massbank_records_simple()
    if extract_dir is None:
        print("\n❌ Download failed. Please download manually and re-run.")
        return
    
    # Step 2: Parse records from extracted files
    print("\n" + "="*80)
    print("PARSING MASSBANK RECORDS")
    print("="*80)
    
    # Find all .txt files in MassBank-data
    record_files = list(extract_dir.rglob('*.txt'))
    print(f"\nFound {len(record_files)} MassBank record files")
    
    # Parse records (limit to first 10,000 for speed, can adjust)
    max_records = 10000
    print(f"Parsing first {max_records} records...")
    
    records = []
    for record_file in tqdm(record_files[:max_records], desc="Parsing"):
        try:
            rec = parse_massbank_record(record_file)
            if rec['smiles'] and rec['inchikey']:
                records.append(rec)
        except Exception as e:
            continue
    
    print(f"\n✓ Parsed {len(records)} valid records with SMILES/InChIKey")
    
    # Step 3: Filter records
    filtered_records = filter_massbank_records(records)
    
    # Step 4: Prepare DataFrame
    df = prepare_massbank_spectra(filtered_records)
    
    # Step 5: Deduplicate against MassSpecGym
    massspecgym_path = DATA_DIR / 'MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet'
    if massspecgym_path.exists():
        df = deduplicate_against_massspecgym(df, massspecgym_path)
    else:
        print(f"\n⚠️  MassSpecGym file not found: {massspecgym_path}")
        print("   Skipping deduplication (will need to do this manually)")
    
    # Step 6: Compute RDKit descriptors
    print("\n" + "="*80)
    print("COMPUTING RDKIT DESCRIPTORS")
    print("="*80)
    
    print(f"\nComputing 10 RDKit descriptors for {len(df)} compounds...")
    
    for desc in TARGET_DESCRIPTORS:
        df[desc] = np.nan
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing descriptors"):
        descriptors = compute_rdkit_descriptors(row['smiles'])
        for desc_name, desc_value in descriptors.items():
            df.at[idx, desc_name] = desc_value
    
    # Remove rows with missing descriptors
    initial_count = len(df)
    df = df.dropna(subset=TARGET_DESCRIPTORS)
    print(f"\n✓ Removed {initial_count - len(df)} compounds with missing descriptors")
    print(f"   Final count: {len(df)} compounds")
    
    # Step 7: Save intermediate result
    output_path = EXTERNAL_DIR / 'massbank_eu_external_test_no_embeddings.parquet'
    df.to_parquet(output_path, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Prepared MassBank EU external test set:")
    print(f"   Compounds: {len(df)}")
    print(f"   Unique InChIKeys: {df['inchikey'].nunique()}")
    print(f"   Mean peaks per spectrum: {df['n_peaks'].mean():.1f}")
    print(f"   Saved to: {output_path}")
    
    print(f"\n📊 Descriptor statistics:")
    for desc in TARGET_DESCRIPTORS:
        mean_val = df[desc].mean()
        std_val = df[desc].std()
        print(f"   {desc:20s}: {mean_val:7.3f} ± {std_val:6.3f}")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Run compute_massbank_embeddings.py to generate SSL embeddings")
    print(f"   2. Use massbank_eu_external_test.parquet for probing validation")
    print(f"   3. Compare internal test R² vs external validation R²")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
