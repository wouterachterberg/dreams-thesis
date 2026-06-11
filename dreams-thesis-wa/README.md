# DreaMS Thesis Subfolder

This directory contains Wouter Achterberg's thesis-specific code, notebooks,
documentation, and tracked lightweight outputs. The top-level `README.md` is the
authoritative reproduction guide; this file is only a local orientation for the
`dreams-thesis-wa/` subtree.

## Current Layout

```text
dreams-thesis-wa/
├── src/                    # data-prep, split, probing, and frozen-baseline helpers
├── scripts/                # final producer scripts, Slurm wrappers, and checks
├── notebooks/              # Axis 1/2 integration and exploratory notebooks
├── results/
│   ├── axis1/
│   │   ├── figures/
│   │   └── indicators/
│   ├── axis2/
│   │   ├── cross_run_integration/
│   │   └── figures/
│   ├── axis3/
│   │   ├── figures/
│   │   └── specific/
│   ├── cross_axis/
│   │   └── figures/
│   ├── holdout/
│   │   └── retrieval/
│   └── shared/
│       └── figures/
└── CODEBASE_MAP.md         # detailed provenance and cleanup audit
```

Retired paths such as `dreams-thesis-wa/figures/`,
`dreams-thesis-wa/results/figures/`, root-level `results/`, and
`vu-cs-research-thesis/` are not part of the current repository layout.

## Current Thesis State

- Splits are the final four-way Murcko scaffold partition: Train, ID-Val, OOD,
  and Holdout. The tracked spectrum-id manifest is
  `dreams-thesis-wa/results/shared/splits.csv`.
- Axis 2 covers 12 conditions:
  ECFP4/MACCS/MAP4 x cosine/BCE x frozen/fine-tuned.
- The fingerprint decoder used for the final frozen/fine-tuned comparison is a
  DeepSets head (`phi -> sum-pool -> rho`), not a single linear layer.
- Axis 3 MAC DDA deployment is complete. The reported model is MACCS-BCE
  fine-tuned, with the frozen MACCS-BCE run used for comparison.
- The reserved Zenodo DOI is `10.5281/zenodo.20598853`.

## Where to Start

- Use the top-level `README.md` for reproduction order and storage policy.
- Use `DATASET_README.md` for Zenodo archive restore instructions.
- Use `CODEBASE_MAP.md` for artifact-to-producer evidence.
- Treat `docs/thesis/THESIS_METHODS_*.md` as archival drafts unless refreshed
  against the current repo state.
