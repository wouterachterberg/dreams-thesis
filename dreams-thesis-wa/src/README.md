# Thesis Source Helpers

This directory holds thesis-specific helper scripts for data preparation,
splitting, probing, embeddings, and frozen-head baselines. The final reproduction
story is documented in the top-level `README.md`; use this file as a map of the
source helpers only.

## Split And Data Preparation Helpers

- `murcko_histogram_splits.py` creates the Murcko-histogram source partition for
  MassSpecGym-derived tables.
- `align_splits_for_finetuning.py` maps that source partition into the final
  four-way thesis split roles:
  - Train: fine-tuning train rows.
  - ID-Val: fine-tuning validation rows.
  - OOD: held-out probing-test rows.
  - Holdout: pristine holdout rows.
- The tracked split manifest is
  `dreams-thesis-wa/results/shared/splits.csv`.
- Large raw/processed data products under `dreams-thesis-wa/data/` remain
  gitignored and are restored from source data or the Zenodo archive.

## Axis 1 Helpers

- Descriptor/probing helpers feed the Axis 1 notebooks and scripts.
- Current tracked Axis 1 outputs live under:
  - `dreams-thesis-wa/results/axis1/indicators/`
  - `dreams-thesis-wa/results/axis1/figures/`

## Axis 2 Helpers

- Fine-tuned runs use DreaMS `FingerprintHead` with the final 12-condition
  matrix: ECFP4/MACCS/MAP4 x cosine/BCE x frozen/fine-tuned.
- Frozen-baseline helpers use the DeepSets fingerprint decoder
  (`phi -> sum-pool -> rho`), matching the final frozen-vs-fine-tuned
  comparison rather than a single linear readout.
- Current tracked Axis 2 summaries and figures live under:
  - `dreams-thesis-wa/results/axis2/cross_run_integration/`
  - `dreams-thesis-wa/results/axis2/figures/`

## Axis 3 And Holdout

Axis 3 and holdout entry points live in `dreams-thesis-wa/scripts/`, not this
directory. Axis 3 MAC DDA deployment is complete and uses MACCS-BCE fine-tuned as
the reported model.
