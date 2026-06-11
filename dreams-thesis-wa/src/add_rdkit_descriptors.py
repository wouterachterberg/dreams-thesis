"""
Add RDKit molecular descriptors to MassSpecGym parquet file.

This script:
1. Loads the parquet file with SSL embeddings
2. Removes old descriptors (mol_weight, logp, subgroups)
3. Adds new RDKit descriptors:
   - AlogP: Wildman-Crippen LogP
   - HBA: Hydrogen Bond Acceptors
   - HBD: Hydrogen Bond Donors
   - TPSA: Topological Polar Surface Area
   - #RotBonds: Number of Rotatable Bonds
   - #AromRings: Number of Aromatic Rings
   - #AliphRings: Number of Aliphatic Rings
   - Fsp3: Fraction of sp3 carbons
   - QED: Quantitative Estimate of Drug-likeness
   - SA: Synthetic Accessibility Score
4. Keeps fingerprints (FPs) unchanged
5. Saves updated parquet file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, QED

# Synthetic Accessibility Score
HAS_SA_SCORER = False
try:
    from rdkit.Chem import RDConfig
    import sys
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SA_SCORER = True
    print("✓ SA_Score module loaded successfully")
except Exception as e:
    print(f"⚠ Warning: SA_Score module not found ({e}). SA scores will be set to NaN.")
    HAS_SA_SCORER = False


def calculate_descriptors(smiles):
    """
    Calculate RDKit descriptors for a single SMILES string.
    
    Returns dict with:
    - alogp: Wildman-Crippen LogP
    - hba: Hydrogen Bond Acceptors
    - hbd: Hydrogen Bond Donors
    - tpsa: Topological Polar Surface Area
    - n_rotatable_bonds: Number of Rotatable Bonds
    - n_aromatic_rings: Number of Aromatic Rings
    - n_aliphatic_rings: Number of Aliphatic Rings
    - fsp3: Fraction of sp3 carbons
    - qed: Quantitative Estimate of Drug-likeness
    - sa_score: Synthetic Accessibility Score
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'alogp': Crippen.MolLogP(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'n_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'n_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'n_aliphatic_rings': Lipinski.NumAliphaticRings(mol),
            'fsp3': Lipinski.FractionCSP3(mol),  # Correct function name
            'qed': QED.qed(mol),
            'sa_score': sascorer.calculateScore(mol) if HAS_SA_SCORER else np.nan
        }
        
        return descriptors
        
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None


def add_descriptors_to_parquet(
    input_path: str,
    output_path: str = None,
    columns_to_remove: list = None,
    force_recalculate: bool = False
):
    """
    Add RDKit descriptors to parquet file.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save output (if None, overwrites input)
        columns_to_remove: List of old/legacy column names to remove if present (default: mol_weight, logp, functional groups)
        force_recalculate: If True, recalculate descriptors even if they exist
    """
    
    # Default: legacy columns that might exist in modified datasets
    # (Safe to specify - will only remove if present)
    if columns_to_remove is None:
        columns_to_remove = [
            'mol_weight', 'logp', 'tpsa',  # Old descriptor names
            'alkene', 'aromatic', 'hydroxyl', 'ketone', 'carboxylic_acid',  # Functional groups
            'amine_primary', 'amide', 'ester', 'nitrile', 'halide', 
            'phosphate', 'thiol', 'nitro'
        ]
    
    print(f"Loading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Check if descriptors already exist
    expected_descriptors = ['alogp', 'hba', 'hbd', 'tpsa', 'n_rotatable_bonds', 
                           'n_aromatic_rings', 'n_aliphatic_rings', 'fsp3', 'qed', 'sa_score']
    existing_descriptors = [col for col in expected_descriptors if col in df.columns]
    
    if existing_descriptors and not force_recalculate:
        print(f"\n✓ Descriptors already exist: {existing_descriptors}")
        print("Skipping descriptor calculation. Use force_recalculate=True to override.")
        
        # Clean up legacy columns if present (backwards compatibility)
        columns_to_drop = [col for col in columns_to_remove if col in df.columns]
        
        # Also remove any 'subgroup' or 'has_' columns (legacy from old pipelines)
        for col in df.columns:
            if col not in existing_descriptors and ('subgroup' in col.lower() or col.startswith('has_')):
                if col not in columns_to_drop:
                    columns_to_drop.append(col)
        
        if columns_to_drop:
            print(f"\n⚠️  Removing legacy columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
            print(f"New shape: {df.shape}")
            
            # Save output
            if output_path is None:
                output_path = input_path
            
            print(f"\nSaving to: {output_path}")
            df.to_parquet(output_path, index=False)
            print("✅ Done!")
        else:
            print("\n✅ No legacy columns found. Dataset is clean.")
        
        return df
    
    # Clean up legacy columns before adding new descriptors (backwards compatibility)
    columns_to_drop = [col for col in columns_to_remove if col in df.columns]
    
    # Also remove any 'subgroup' or 'has_' columns (legacy from old pipelines)
    for col in df.columns:
        if 'subgroup' in col.lower() or col.startswith('has_'):
            if col not in columns_to_drop:
                columns_to_drop.append(col)
    
    if columns_to_drop:
        print(f"\n⚠️  Removing legacy columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    else:
        print("\n✓ No legacy columns found (fresh dataset)")

    
    # Calculate new descriptors
    print("\nCalculating RDKit descriptors...")
    
    descriptor_results = []
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        smiles = row.get('smiles', None)
        
        if pd.isna(smiles) or smiles == '':
            # Add NaN for all descriptors if SMILES is missing
            descriptor_results.append({
                'alogp': np.nan,
                'hba': np.nan,
                'hbd': np.nan,
                'tpsa': np.nan,
                'n_rotatable_bonds': np.nan,
                'n_aromatic_rings': np.nan,
                'n_aliphatic_rings': np.nan,
                'fsp3': np.nan,
                'qed': np.nan,
                'sa_score': np.nan
            })
            failed_count += 1
        else:
            result = calculate_descriptors(smiles)
            if result is None:
                # Add NaN for all descriptors if calculation failed
                descriptor_results.append({
                    'alogp': np.nan,
                    'hba': np.nan,
                    'hbd': np.nan,
                    'tpsa': np.nan,
                    'n_rotatable_bonds': np.nan,
                    'n_aromatic_rings': np.nan,
                    'n_aliphatic_rings': np.nan,
                    'fsp3': np.nan,
                    'qed': np.nan,
                    'sa_score': np.nan
                })
                failed_count += 1
            else:
                descriptor_results.append(result)
    
    # Add descriptors to dataframe
    descriptor_df = pd.DataFrame(descriptor_results)
    
    # Drop any existing descriptor columns that would be duplicated
    existing_desc_cols = [col for col in descriptor_df.columns if col in df.columns]
    if existing_desc_cols:
        print(f"\nRemoving existing descriptor columns before re-adding: {existing_desc_cols}")
        df = df.drop(columns=existing_desc_cols)
    
    df = pd.concat([df, descriptor_df], axis=1)
    
    print(f"\nNew shape: {df.shape}")
    print(f"Failed molecules: {failed_count}/{len(df)} ({failed_count/len(df)*100:.2f}%)")
    print(f"\nNew columns added: {descriptor_df.columns.tolist()}")
    
    # Print descriptor statistics
    print("\nDescriptor Statistics:")
    print("=" * 60)
    for col in descriptor_df.columns:
        if col in df.columns:
            valid_count = int(df[col].notna().sum())
            mean_val = float(df[col].mean()) if df[col].notna().any() else np.nan
            std_val = float(df[col].std()) if df[col].notna().any() else np.nan
            print(f"{col:20s}: mean={mean_val:8.3f}, std={std_val:8.3f}, valid={valid_count}/{len(df)}")
    print("=" * 60)
    
    # Save output
    if output_path is None:
        output_path = input_path
    
    print(f"\nSaving to: {output_path}")
    df.to_parquet(output_path, index=False)
    print("✅ Done!")
    
    return df


if __name__ == "__main__":
    # Path to your parquet file
    input_file = "../data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings.parquet"
    
    # You can optionally save to a different file (or set to None to overwrite)
    output_file = input_file  # Overwrites original
    # output_file = "../data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings_updated.parquet"
    
    # Run the descriptor addition
    df = add_descriptors_to_parquet(
        input_path=input_file,
        output_path=output_file
    )
    
    print("\n" + "=" * 60)
    print("FINAL COLUMNS:")
    print("=" * 60)
    print(df.columns.tolist())
