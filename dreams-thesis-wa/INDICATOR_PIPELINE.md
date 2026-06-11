# Indicator Analysis Pipeline

**Overview**: Three-stage pipeline to compute SSL embedding quality indicators using nearest-neighbor analysis.

## Data Flow

```
TASK 0 (Setup)
    ↓ [reads probing_test.parquet + all_rdkit_descriptors.parquet]
    ↓ [normalizes embeddings, merges on 'smiles']
    ↓ [saves indicator_data.pkl]
    ↓
    ├── indicator_data.pkl  (48 MB)
    │   ├── embeddings: (45,185 × 1024)
    │   ├── descriptors: (45,185 × 208)
    │   ├── descriptor_names: list of 208 names
    │   ├── spectrum_to_molecule: array of InChIKeys
    │   ├── molecule_to_spectra: dict InChIKey → spectrum indices
    │   └── ... [metadata]
    │
    ├──────────────────────────────────────────┐
    │                                          │
    TASK 1 (Build kNN Graphs)          [cached in memory]
        ↓ [reads indicator_data.pkl]
        ↓ [computes cosine-distance kNN for k ∈ {10, 50, 100}]
        ↓ [filters exclusive version: removes same-molecule neighbors]
        ↓ [saves 6 kNN graphs]
        │
        ├── knn_k10_inclusive.pkl  (2.1 MB)  ← indices, distances, similarities
        ├── knn_k10_exclusive.pkl  (2.1 MB)
        ├── knn_k50_inclusive.pkl  (10.5 MB)
        ├── knn_k50_exclusive.pkl  (10.5 MB)
        ├── knn_k100_inclusive.pkl (21 MB)
        └── knn_k100_exclusive.pkl (21 MB)
        │
        └──────────────────────────────────────────┐
                                                   │
        TASK 2 (Indicator 1: NN Descriptor Consistency)
            ↓ [reads indicator_data.pkl]
            ↓ [reads knn_k{10,50,100}_{inclusive,exclusive}.pkl]
            ↓ [computes for each descriptor:]
            ↓   - mean |Δdescriptor| for NN pairs
            ↓   - mean |Δdescriptor| for random pairs
            ↓   - effect_ratio = random_mean / nn_mean
            ↓   - Spearman(embedding_similarity, -|Δdescriptor|)
            │
            └── nn_descriptor_consistency.csv
                ├── Rows: 208 descriptors × 3 k values × 2 neighbor types
                ├── Columns: descriptor, k, neighbors, nn_mean_diff, 
                │            random_mean_diff, effect_ratio, spearman_corr
                └── Summary: Top/Bottom 10 by effect_ratio (k=50, exclusive)

        TASK 3 (Descriptor Family Aggregation)
            ↓ [reads nn_descriptor_consistency.csv]
            ↓ [classifies descriptors into families]
            ↓ [aggregates family-level statistics]
            │
            └── descriptor_family_summary.csv

        TASK 4 (Merge Probe & Indicator Results)
            ↓ [reads all_descriptors_probing_results.csv]
            ↓ [reads nn_descriptor_consistency.csv (k=50, exclusive)]
            ↓ [merges on descriptor name]
            ↓ [computes MLP gain = R²_MLP - R²_linear]
            ↓ [flags: high-linear, non-linear, low-probe-but-strong-indicator]
            │
            └── probe_indicator_merged.csv
                ├── Columns: descriptor, r2_linear, r2_mlp, mlp_gain,
                │            effect_ratio, spearman_corr, flags
                └── Enables: probe vs indicator scatter plots, anomaly detection

        TASK 5 (Main Figure: Indicator vs Probe)
            ↓ [reads probe_indicator_merged.csv]
            ↓ [creates 2-panel figure]
            │   Panel A: R² vs effect_ratio scatter (colored by MLP gain)
            │   Panel B: NN vs random |Δ| for representative descriptors
            │
            └── dreams-thesis-wa/results/axis1/figures/
                  ├── indicator_vs_probe.png (300 DPI)
                  └── indicator_vs_probe.pdf (vector)

        TASK 6 (Interpretation Summary)
            ↓ [reads probe_indicator_merged.csv]
            ↓ [reads descriptor_family_summary.csv]
            ↓ [generates interpretation bullets]
            │
            └── indicator_summary.txt
                  ├── 3-5 bullet points for thesis
                  ├── Summary statistics
                  └── Notable examples

        TASK 7 (Additional Figures & Analysis)                [future]
            ↓ [reads all indicator CSVs + family summaries]
            │
            └── dreams-thesis-wa/results/axis1/figures/
                  ├── descriptor_family_heatmap.pdf
                  ├── spearman_vs_ratio_scatter.pdf
                  └── ...
```

## Key Cached Artifacts

### 1. `indicator_data.pkl` (TASK 0)
- **Size**: ~48 MB
- **Contents**:
  - `embeddings`: (45,185, 1024) float32 array — SSL embeddings
  - `descriptors`: (45,185, 208) float32 array — RDKit descriptors
  - `descriptor_names`: list of 208 descriptor names
  - `spectrum_to_molecule`: (45,185,) array of InChIKeys
  - `molecule_to_spectra`: dict mapping InChIKey → list of spectrum indices
  - `inchikeys`, `smiles`, metadata (n_spectra, n_molecules, etc.)

### 2. kNN Graphs (TASK 1)
**6 files total **:
- **Inclusive** (k=10, 50, 100): All k nearest neighbors
  - `knn_k{k}_inclusive.pkl`: {indices, distances, similarities, metadata}
- **Exclusive** (k=10, 50, 100): Neighbors from different molecules only
  - `knn_k{k}_exclusive.pkl`: {indices (with -1 padding), distances (NaN padded), similarities, metadata}

**Sizes**:
- k=10: ~2.1 MB each (inclusive + exclusive)
- k=50: ~10.5 MB each
- k=100: ~21 MB each

### 3. Results CSV Files

#### `nn_descriptor_consistency.csv` (TASK 2)
- **Rows**: 208 descriptors × 3 k values × 2 neighbor types = 1,248 rows
- **Columns**:
  - `descriptor`: descriptor name
  - `k`: neighborhood size (10, 50, 100)
  - `neighbors`: 'inclusive' or 'exclusive'
  - `nn_mean_diff`: mean |Δdescriptor| in kNN
  - `random_mean_diff`: mean |Δdescriptor| in random pairs
  - `effect_ratio`: random_mean / nn_mean (higher = more consistent)
  - `spearman_corr`: correlation between embedding similarity and descriptor agreement

#### `descriptor_family_summary.csv` (TASK 3)
- **Rows**: ~10-15 descriptor families
- **Columns**:
  - `family`: family name (Topological, Surface Area, Fragment Counts, etc.)
  - `n_descriptors`: number of descriptors in family
  - `mean_effect_ratio`: average effect ratio across family
  - `mean_spearman`: average Spearman correlation
  - Family-level statistics for interpretability

#### `probe_indicator_merged.csv` (TASK 4)
- **Rows**: 208 descriptors
- **Columns**:
  - `descriptor`: descriptor name
  - `r2_linear`: linear probe R² (test set)
  - `r2_mlp`: MLP probe R² (test set)
  - `mlp_gain`: R²_MLP - R²_linear (non-linearity measure)
  - `effect_ratio`: from TASK 2 (k=50, exclusive)
  - `spearman_corr`: from TASK 2 (k=50, exclusive)
  - `flag_high_linear`: R²_linear > 0.3
  - `flag_nonlinear`: MLP gain > 0.05
  - `flag_low_probe_strong_indicator`: low R² but high effect_ratio
- **Purpose**: Unified table for probe vs indicator comparison

## Execution Sequence

```bash
# First run setup
jupyter notebook task0_indicator_setup.ipynb
# Run all cells (TASK 0)

# Then build kNN graphs once
jupyter notebook task1_build_knn_graphs.ipynb
# Run all cells (TASK 1)

# Run indicator analyses
jupyter notebook task2_indicator1_nn_consistency.ipynb
# Run all cells (TASK 2) — may take 30-60 minutes

jupyter notebook task3_descriptor_families.ipynb
# Run all cells (TASK 3) — ~10 seconds

# Merge with probe results
jupyter notebook task4_merge_probe_indicator.ipynb
# Run all cells (TASK 4) — requires probe results CSV

# Generate main figure
jupyter notebook task5_indicator_vs_probe_figure.ipynb
# Run all cells (TASK 5) — creates PNG + PDF figures

# Generate interpretation summary
jupyter notebook task6_interpretation_summary.ipynb
# Run all cells (TASK 6) — creates text summary for thesis

# Additional analyses and figures
# jupyter notebook task7_additional_figures.ipynb  [future]
```

## Key Design Decisions

1. **Separate kNN computation** (TASK 1):
   - Compute once, reuse across all descriptors
   - Avoid redundant cosine distance calculations

2. **Two neighbor versions**:
   - **Inclusive**: all k neighbors (sanity check)
   - **Exclusive**: filter same-molecule neighbors (real analysis)
   - Both stored for comparison

3. **Effect ratio = random/NN**:
   - Ratio > 1: descriptor is consistent in embedding space
   - Ratio < 1: descriptor is inconsistent

4. **Spearman correlation**:
   - Correlates embedding similarity with descriptor agreement
   - Validates that geometry captures descriptor structure

## Validation Checklist

- [x] TASK 0: All 45,185 spectra load correctly
- [x] TASK 0: All 208 descriptors present (no NaN values > threshold)
- [x] TASK 0: indicator_data.pkl saved (~48 MB)
- [x] TASK 1: 6 kNN graph caches saved (~68 MB total)
- [x] TASK 1: Exclusive graphs have valid-neighbor statistics logged
- [x] TASK 2: CSV has 1,248 rows (208 × 3 × 2)
- [x] TASK 2: effect_ratio values are positive (random_mean >> 0)
- [x] TASK 2: Top 10 descriptors have ratio > 1 (consistent)
- [x] TASK 2: Summary printed to console
- [x] TASK 2: Spearman sign corrected (positive values for best descriptors)
- [x] TASK 3: Descriptor families classified and aggregated
- [x] TASK 3: Family summary CSV saved
- [ ] TASK 4: Probe results loaded successfully
- [ ] TASK 4: Merge successful (208 descriptors with both metrics)
- [ ] TASK 4: probe_indicator_merged.csv saved
- [ ] TASK 4: Correlation analysis complete (R² vs effect_ratio)
- [ ] TASK 5: Figure generated with 2 panels (scatter + bars)
- [ ] TASK 5: PNG saved (300 DPI) to dreams-thesis-wa/results/axis1/figures/
- [ ] TASK 5: PDF saved (vector) to dreams-thesis-wa/results/axis1/figures/
- [ ] TASK 5: Representative descriptors selected and visualized
- [ ] TASK 5: Correlation coefficient computed and displayed
- [ ] TASK 6: Interpretation bullets generated (3-5 points)
- [ ] TASK 6: Summary text file saved to indicators/
- [ ] TASK 6: Key patterns identified (strongest, anomalies, disagreements)
- [ ] TASK 6: Cautious language used (no causal claims)
