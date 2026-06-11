"""
Fetch MassBank EU records via REST API

Simpler alternative to bulk download - queries MassBank API directly.
This is slower but easier to set up.

Usage:
    python src/fetch_massbank_api.py --max-compounds 500
"""

import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
import argparse

# MassBank EU API
MASSBANK_API = "https://massbank.eu/MassBank/api/v2"

# Save directory
EXTERNAL_DIR = Path(__file__).parent.parent / 'data' / 'external' / 'massbank_eu'
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def search_massbank_records(ion_mode='POSITIVE', max_results=1000):
    """
    Search MassBank for records matching criteria.
    
    Returns list of accession IDs.
    """
    print(f"🔍 Searching MassBank for {ion_mode} mode spectra...")
    
    # Simple query for positive mode
    url = f"{MASSBANK_API}/spectra"
    params = {
        'ionMode': ion_mode,
        'ms': 2  # MS/MS spectra only
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        results = response.json()
        accessions = [rec['accession'] for rec in results[:max_results]]
        
        print(f"   Found {len(accessions)} records")
        return accessions
        
    except Exception as e:
        print(f"   ❌ API search failed: {e}")
        return []


def fetch_record_details(accession: str):
    """Fetch detailed record for a given accession ID"""
    url = f"{MASSBANK_API}/spectrum/{accession}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None


def parse_massbank_api_record(record: dict):
    """Parse JSON record from MassBank API"""
    try:
        # Extract key fields
        metadata = record.get('metadata', {})
        spectrum = record.get('spectrum', [])
        
        return {
            'accession': record.get('accession'),
            'smiles': metadata.get('smiles'),
            'inchikey': metadata.get('inchikey'),
            'formula': metadata.get('formula'),
            'precursor_mz': metadata.get('precursorMz'),
            'collision_energy': metadata.get('collisionEnergy'),
            'instrument_type': metadata.get('instrumentType'),
            'ion_mode': metadata.get('ionMode'),
            'mzs': [peak['mz'] for peak in spectrum],
            'intensities': [peak['intensity'] for peak in spectrum]
        }
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Fetch MassBank EU external test set')
    parser.add_argument('--max-compounds', type=int, default=500,
                        help='Maximum number of compounds to fetch')
    args = parser.parse_args()
    
    print("="*80)
    print("MASSBANK EU FETCH VIA API")
    print("="*80)
    print(f"\nTarget: {args.max_compounds} unique compounds")
    print(f"Criteria: Positive ESI, MS/MS spectra")
    
    # Search for records
    accessions = search_massbank_records(max_results=args.max_compounds * 3)
    
    if not accessions:
        print("\n❌ No records found. Try manual download approach instead.")
        return
    
    # Fetch detailed records
    print(f"\n📥 Fetching details for {len(accessions)} records...")
    
    records = []
    for acc in tqdm(accessions, desc="Fetching"):
        record = fetch_record_details(acc)
        if record:
            parsed = parse_massbank_api_record(record)
            if parsed and parsed['smiles'] and parsed['inchikey']:
                records.append(parsed)
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"\n✅ Fetched {len(records)} valid records")
    
    # Save raw records
    output_path = EXTERNAL_DIR / 'massbank_raw_records.json'
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"   Saved to: {output_path}")
    print(f"\n💡 Next: Run prepare_massbank_external_test.py to process these records")


if __name__ == '__main__':
    main()
