

# DreaMS thesis reproducibility repository

[![DOI](assets/doi.svg)](https://doi.org/10.5281/zenodo.20598853)

<p align="center">
  <img src="assets/from-spectra-to-structure-dark.png" alt="From Spectra to Structure" width="100%">
</p>

This repository contains the code, result artifacts, and provenance notes for
Wouter Achterberg's thesis experiments on DreaMS representations for MS/MS
analysis. It documents the repository layout and the steps needed to reproduce
the reported results.

The code is derived from DreaMS by pluskal-lab, published in Nature
Biotechnology in 2025 and released under the MIT License. The upstream DreaMS
package code remains in `dreams/`; thesis-specific work is kept under
`dreams-thesis-wa/`, with release metadata and shared assets at the repository
root.

## Repository structure

```text
dreams/                         upstream DreaMS package code; do not refactor for cleanup
dreams-thesis-wa/
  CODEBASE_MAP.md               full local audit and trace map
  src/                          thesis data-prep, splits, embedding, probing helpers
  scripts/                      Slurm wrappers and final figure/result scripts
  notebooks/                    analysis notebooks used for Axis 1/2 and integration
  results/
    axis1/                      tracked Axis 1 figures and indicator summaries
    axis2/                      tracked Axis 2 figures and integration summaries
    axis3/                      tracked Axis 3 reports, summaries, and figures
    cross_axis/                 tracked canonical cross-axis outputs
    holdout/                    tracked MACCS-BCE holdout headline metrics/ranks
    shared/                     shared dataset-level figures and split manifest
docs/                           inherited DreaMS documentation
assets/, tutorials/, tests/     inherited DreaMS assets, tutorials, and tests
```

Large local caches, predictions, checkpoints, raw data, HDF5/parquet files, and
NumPy arrays are intentionally ignored.

## Data access

MassSpecGym is the public benchmark used for Axis 1/2 spectra and molecular
structure data. Cite the benchmark as:

```bibtex
@article{bushuiev2024massspecgym,
  title={MassSpecGym: A benchmark for the discovery and identification of molecules},
  author={Bushuiev, Roman and others},
  journal={arXiv preprint arXiv:2410.23326},
  year={2024},
  doi={10.48550/arXiv.2410.23326},
  url={https://arxiv.org/abs/2410.23326}
}
```

The MAC Axis 3 dataset is controlled-access local lab data. The raw mzML and
identity/fragment CSV files are not committed because they are not public
redistribution material and contain experiment-specific metadata. Request access
from the thesis author or the MAC dataset owner before attempting a full Axis 3
replay.

## Environment

The evaluation environment is Python 3.10, with the packages pinned in [requirements.txt](requirements.txt) (captured 2026-06-05).

Recommended setup:

```bash
conda env create -f environment.yml
conda activate dreams-thesis-wa
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The main thesis seed is `3407` for training/evaluation scripts and Axis 3
preparation. Some older split/probing helper code uses seed `42`; see
`CODEBASE_MAP.md` for the exact evidence.

## Reproduction order

The full per-notebook dependency chain is in
`dreams-thesis-wa/CODEBASE_MAP.md` (section 7). The five stages run in order:

1. **Axis 1 (probing):** MassSpecGym spectra -> SSL embeddings -> ridge/MLP probes over 201 RDKit descriptors -> `probe_indicator_merged.csv` and the indicator figure.
2. **Axis 2 (fingerprint prediction):** split HDF5 + pretrained DreaMS -> the 12 fine-tuned/frozen runs -> per-run predictions -> cross-run tables and `axis2_publication_figures.py`.
3. **Cross-axis bridge:** Axis 1 probe R^2 + Axis 2 per-bit AUROC -> `cross_axis_bridge.py`.
4. **Axis 3 (external deployment):** MAC DDA spectra + Axis 2 references -> `run_axis3_tier1_results.py` with the MACCS-BCE fine-tuned and frozen runs.
5. **Holdout:** the reserved Holdout split + the MACCS-BCE checkpoint -> `evaluate_holdout_maccs_bce.py`.

You do not need to re-execute the notebooks to inspect the reported outputs; the
lightweight public result artifacts are committed, and the larger caches and
checkpoints are restored through `DATASET_README.md`.

## Storage policy

Code, notebooks, documentation, and the lightweight canonical result artifacts (Axis 1-3 figures and summaries, cross-axis outputs, holdout metrics, and the `splits.csv` fold manifest) are committed to git. The large caches - the per-model prediction bundle, trained checkpoints, the RDKit descriptor matrix, and raw/processed datasets - are too big for git: the released ones are archived on Zenodo (see `DATASET_README.md`), and the rest stay on Snellius/local storage with their original data sources.

## Citation

Use [CITATION.cff](CITATION.cff) for this thesis repository. Also cite the
upstream DreaMS paper and MassSpecGym when reusing the method or data.
