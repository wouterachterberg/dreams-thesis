# Legacy quarantine

This directory contains tracked files that CODEBASE_MAP section 6 identified as
dead, superseded, dropped, or one-off scratch code. They were moved here in
Phase 2 with `git mv` so history is preserved and the quarantine is reversible.

Before moving these files, the canonical producer set was searched for direct
references to the original filenames. No references were found in the current
canonical scripts/notebooks.

| Legacy path | Original path | Why quarantined | Superseded by |
|---|---|---|---|
| `dreams-thesis-wa/legacy/notebooks/per_bit_morgan_analysis.ipynb` | `dreams-thesis-wa/notebooks/per_bit_morgan_analysis.ipynb` | First Morgan/ECFP4 cosine analysis; writes old `results/per_bit_analysis` outputs. | `dreams-thesis-wa/notebooks/model_agnostic_eval_artifact_builder.ipynb` for all 12 Axis 2 conditions. |
| `dreams-thesis-wa/legacy/notebooks/debugging_local_results.ipynb` | `dreams-thesis-wa/notebooks/debugging_local_results.ipynb` | Local scratch/debugging notebook with no canonical role found. | No canonical replacement; retained for audit only. |
| `dreams-thesis-wa/legacy/notebooks/axis_2_analysis_summary.ipynb` | `dreams-thesis-wa/notebooks/axis_2_analysis_summary.ipynb` | Legacy summary notebook reading old `results/per_bit_analysis` outputs. | `dreams-thesis-wa/notebooks/cross_run_integration_master.ipynb` and `dreams-thesis-wa/scripts/axis2_publication_figures.py`. |
| `dreams-thesis-wa/legacy/notebooks/DreaMS.code-workspace` | `dreams-thesis-wa/notebooks/DreaMS.code-workspace` | Editor-local workspace file with nonportable external path references. | Repository root plus documented paths in `README.md`. |
| `dreams-thesis-wa/legacy/scripts/run_retrieval_evaluation.py` | `run_retrieval_evaluation.py` | Hardcoded old local checkpoint/output paths and legacy retrieval workflow. | `dreams-thesis-wa/notebooks/model_agnostic_eval_artifact_builder.ipynb` per-run retrieval artifacts. |
| `dreams-thesis-wa/legacy/scripts/partition_raw_mzml.py` | `dreams-thesis-wa/scripts/partition_raw_mzml.py` | Dropped Tier 2 recovery-from-raw helper. | Tier 1 Axis 3 prepared dataset path via `prepare_axis3_tier1_dataset.py` and `run_axis3_tier1_results.py`. |
| `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` | `fix_spearman_sign.py` | One-off post-hoc Spearman table mutation; creates provenance ambiguity. | Planned single-source fix in `cross_axis_bridge.py` if Phase 5 is approved. |

Phase 2 also requested root `bottom_20_descriptors.png` and
`axis3_umap_figures.py` if present. Neither is tracked in this cleanup worktree,
so no move was performed for those candidates.
