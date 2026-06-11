# CODEBASE MAP

Generated: 2026-06-04. Scope: local filesystem and read-only git inspection only. I did not execute notebooks, training, evaluation, install commands, or mutating git commands. I did not read checkpoint tensor contents; checkpoint identity is based on size, inode/link metadata, and source code evidence unless noted.

Status vocabulary used below:

- FULLY-TRACED: producer, inputs, output writer, and downstream consumer are identified.
- PARTIALLY-TRACED: most of the chain is identified, but a checkpoint/config/notebook execution gap remains.
- ORPHAN: artifact exists or is consumed, but no producer was found in this repo.
- AMBIGUOUS: more than one plausible producer writes the same artifact or run layout.
- UNKNOWN - needs Wouter: local evidence is insufficient.

## 1. Repo overview

### Inherited from DreaMS upstream

- `dreams/`: upstream package for DreaMS models, data formats, model heads, IO, LC-MS utilities, and training entry points.
- `docs/`, `tutorials/`, `experiments/`, `assets/`, `tests/`: upstream documentation, examples, figures, and tests.
- `README.md`, `LICENSE`, `setup.py`, `uv.lock`: upstream project metadata. `setup.py` declares Python `>=3.11` and the main pinned package set.

### Modified upstream files

Compared with root commit `e359b1ae2feaebc8dfda042f1a9f74437384648b`, these files in `dreams/` changed. This identifies thesis/fork modifications relative to the earliest local commit, not a verified upstream pluskal-lab merge-base.

| File | Evidence | One-line note |
|---|---:|---|
| `dreams/algorithms/lsh/__init__.py`, `dreams/algorithms/lsh/lsh.py` | `git diff --name-status e359... HEAD -- dreams` | LSH package changed; likely inherited fork work, not directly tied to thesis outputs. |
| `dreams/algorithms/murcko_hist/*` | same | Murcko histogram split code changed; relevant to dataset split lineage. |
| `dreams/api.py` | same | API expanded/changed; used by embedding and inference helpers. |
| `dreams/cli.py` | same | Added CLI entry point. |
| `dreams/definitions.py` | same | Path/export definitions changed; used by Snellius wrappers via `export()`. |
| `dreams/models/dreams/dreams.py` | same | DreaMS model changed; affects fine-tuned checkpoints. |
| `dreams/models/heads/heads.py` | `dreams/models/heads/heads.py:574`, `:579`, `:622`, `:679` | Fingerprint head supports `fp_maccs_166`, `fp_map4_2048`, BCE loss, `pos_weight`, and OneCycleLR. |
| `dreams/training/train.py` | `dreams/training/train.py:214-221`, `:265-266`, `:350` | Main fine-tuning entry point for DreaMS fingerprint heads. |
| `dreams/training/train_argparse.py` | `dreams/training/train_argparse.py:70-75`, `:130` | Adds `--fp_loss`, `--fp_pos_weight`, scheduler flag, and checkpoint count args. |
| `dreams/utils/data.py`, `dformats.py`, `io.py`, `lcms.py`, `misc.py`, `mols.py`, `plots.py`, `spectra.py` | diff list | Utilities changed; relevant where called by thesis scripts. |
| `dreams/training/*_karolina.sh` | diff list | Deleted old upstream/fork wrappers. |

### Thesis additions

- `dreams-thesis-wa/src/`: data prep, fingerprint cache, SSL embedding, probing, split, and frozen-baseline code.
- `dreams-thesis-wa/scripts/`: Slurm wrappers, publication figure scripts, Axis 3 deployment, holdout evaluation, and one-off utilities.
- `dreams-thesis-wa/notebooks/`: Axis 1 indicator/probing notebooks, Axis 2 metric builders, cross-run integration, frozen baseline notebook, and exploratory notebooks.
- `dreams-thesis-wa/results/`: generated thesis results, organised by axis under `axis1/`, `axis2/`, `axis3/`, `cross_axis/`, `holdout/`, and `shared/`.
- `docs/thesis/THESIS_METHODS_*.md`: methods drafts/snippets/tables/verification notes. They overlap with each other but have different roles: extraction narrative, code snippets, quick-reference tables, and verification checklist.
- `docs/thesis/PROBING_TEST_EXPLANATION.md`: split/probing rationale.
- Max's inherited memory-optimisation and CANOPUS-prep tooling was removed during repository cleanup; see tag v1.0-thesis for the as-run snapshot.
- Retired cleanup-era folders such as root-level `results/` and `vu-cs-research-thesis/` are not part of the current repository layout.

## 2. Per-axis file map

### Axis 1 - representation probing and indicators

| Path | Role | Purpose | Inputs | Outputs | Evidence |
|---|---|---|---|---|---|
| `dreams-thesis-wa/src/generate_ssl_embeddings.py` | entry/helper | Generates SSL embeddings for MassSpecGym-style tables. | processed spectra/HDF5/parquet, pretrained DreaMS. | embedding files under thesis data. | README calls this Step 1. |
| `dreams-thesis-wa/src/add_rdkit_descriptors.py` | entry/helper | Adds RDKit descriptors to processed spectra/molecules. | processed MassSpecGym data. | descriptor-enriched data. | README Step 2. |
| `dreams-thesis-wa/notebooks/probe_ssl_embeddings.ipynb` | notebook entry | Original probing notebook. | SSL embeddings, descriptors. | `results/probing_results_ssl.pkl`. | writer at `probe_ssl_embeddings.ipynb:1236`. |
| `dreams-thesis-wa/notebooks/exploratory/probe_all_rdkit_descriptors.ipynb` | notebook entry | Full 201-descriptor linear/MLP probing. | SSL embeddings, RDKit descriptor table. | `results/all_descriptors_probing_results*.{csv,pkl}`, `top50_descriptors_probing.csv`. | writers at `probe_all_rdkit_descriptors.ipynb:722`, `:783`, `:1061`, `:1073`, `:1079`. |
| `dreams-thesis-wa/notebooks/task0_indicator_setup.ipynb` | notebook entry | Builds indicator cache. | Axis 1 descriptor/probe tables. | `dreams-thesis-wa/results/axis1/indicators/indicator_data.pkl`, setup summary. | `task0_indicator_setup.ipynb:559`, `:604`, `:647`. |
| `dreams-thesis-wa/notebooks/task1_build_knn_graphs.ipynb` | notebook entry | KNN graph cache for nearest-neighbour indicator. | indicator cache. | `knn_k{10,50,100}_{inclusive,exclusive}.pkl`, stats. | `task1_build_knn_graphs.ipynb:414`, `:421`, `:559`. |
| `dreams-thesis-wa/notebooks/task2_indicator1_nn_consistency.ipynb` | notebook entry | Nearest-neighbour descriptor consistency. | KNN graphs, descriptor/probe data. | `nn_descriptor_consistency.csv`. | `task2_indicator1_nn_consistency.ipynb:467`. |
| `dreams-thesis-wa/notebooks/task3_descriptor_families.ipynb` | notebook entry | Descriptor-family summaries. | descriptor/probe tables. | `descriptor_family_summary.csv`. | `task3_descriptor_families.ipynb:401`. |
| `dreams-thesis-wa/notebooks/task4_merge_probe_indicator.ipynb` | notebook entry | Merges probing and indicator tables. | probe results, indicator summaries. | `dreams-thesis-wa/results/axis1/indicators/probe_indicator_merged.csv`. | `task4_merge_probe_indicator.ipynb:503-504`. |
| `dreams-thesis-wa/notebooks/task5_indicator_vs_probe_figure.ipynb` | notebook entry | Main Axis 1 scatter/figure. | `probe_indicator_merged.csv`. | `indicator_vs_probe.{png,pdf}`, `scatter_standalone.{png,pdf}`. | `task5_indicator_vs_probe_figure.ipynb:512-518`, `:778-783`. |
| `dreams-thesis-wa/scripts/create_superclass_heatmap.py` | script entry | NPClassifier superclass distribution figure. | `finetuning.hdf5`, superclass cache/table. | `superclass_distribution_full.csv`, `figures/superclass_heatmap.pdf`. | `create_superclass_heatmap.py:91`, `:148`. |

### Axis 2 - fingerprint prediction

| Path | Role | Purpose | Inputs | Outputs | Evidence |
|---|---|---|---|---|---|
| `dreams/training/train.py` | entry | Fine-tunes DreaMS backbone with `FingerprintHead`; not frozen baseline training. | `finetuning.hdf5`, pretrained SSL checkpoint, CLI args. | Lightning checkpoints. | `train.py:214-221`, `:265-266`, `:350`. |
| `dreams-thesis-wa/scripts/fine_tune_test.sh` | Slurm entry | Historical base Snellius wrapper for the fine-tuned half of the final 12-condition matrix. | `$HOME/DreaMS/.../finetuning.hdf5`, `${PRETRAINED}/ssl_model.ckpt`. | `$HOME/DreaMS/dreams-thesis-wa/results/finetuning/$RUN_NAME`. | `fine_tune_test.sh:41`, `:47`, `:59`, `:126-145`. |
| `dreams-thesis-wa/scripts/fine_tune_test_{bce,maccs,maccs_bce,map4,map4_bce}.sh` | thin wrappers | Select fingerprint and loss via env vars. | same as base wrapper. | same as base wrapper. | wrapper `exec bash "$HOME/DreaMS/.../fine_tune_test.sh"` lines 13-17. |
| `dreams-thesis-wa/scripts/fine_tune_round2.sh` | Slurm entry | Round 2 BCE fine-tuning with fp64 and OneCycleLR. | same. | `results/finetuning/<run>`. | `fine_tune_round2.sh:177-193`, `:220`. |
| `dreams-thesis-wa/notebooks/frozen_embedding_deepsets_baselines.ipynb` | notebook entry | Trains the six frozen DeepSets baseline checkpoint dirs, i.e. the frozen half of the final 12-condition matrix. | frozen SSL embeddings HDF5, fingerprint targets. | `dreams-thesis-wa/results/frozen_deepsets_baselines/frozen_*`, `frozen_baseline_summary.csv`. | tags at `:219-225`; config at `:77-80`, `:250`, `:442-445`; summary at `:679-680`. |
| `dreams-thesis-wa/src/frozen_allpeaks_baselines.py` | script entry | Script version for frozen all-peaks DeepSets baseline. | `finetuning_with_ssl_embeddings.hdf5`, `finetuning.hdf5`, `fingerprint_cache.npz`. | `dreams-thesis-wa/results/frozen_allpeaks_baselines/<run_tag>` by current code. | docstring `:6-17`, args `:61-67`, checkpoint saves `:380-381`, config `:416`. |
| `dreams-thesis-wa/src/frozen_allpeaks_inference.py` | script entry | Exports frozen head predictions/targets into `model_runs`. | frozen checkpoints, VAL/OOD data. | `model_runs/<run_tag>/axis2_artifacts/*.npy`, `run_config.json`. | outputs `:437-441`, config `:464`. |
| `dreams-thesis-wa/scripts/h100_frozen_allpeaks_pipeline.sh` | Slurm wrapper | Current H100 end-to-end frozen all-peaks pipeline. | HDF5 data, fingerprint cache, DreaMS repo. | persistent `model_runs` and `frozen_allpeaks_baselines`. | persistent roots `:36-37`, train tags `:236-241`, inference `:244-256`. |
| `dreams-thesis-wa/scripts/h100_batch_inference.py` | script entry | Batch inference for the six fine-tuned runs in the final 12-condition matrix; writes predictions only. | checkpoints, OOD parquet, finetuning HDF5. | `y_pred.npy`, `y_pred_val.npy`, `inference_only_metadata.json`. | specs `:31-79`, outputs `:570-572`, metadata `:673`. |
| `dreams-thesis-wa/scripts/h100_batch_inference.sh` | Slurm wrapper | Stages datasets/checkpoints to scratch, runs H100 inference, syncs to persistent `model_runs`. | Snellius `$HOME/DreaMS`, checkpoint base. | `$REPO_ROOT/dreams-thesis-wa/results/model_runs`. | persistent root `:35`, scratch root `:44`, sync `:180-185`. |
| `dreams-thesis-wa/notebooks/model_agnostic_eval_artifact_builder.ipynb` | notebook entry | Computes per-run AUROC, threshold sweep, retrieval artifacts for all 12 runs. | `model_runs/<tag>/axis2_artifacts/y_pred*.npy`, `y_true*.npy`, checkpoints if cache absent. | `per_bit_auroc`, `threshold_sweep`, `retrieval`, `artifact_manifest.csv`, `run_config.json`. | specs `:138-231`; outputs `:936`, `:996-1005`, `:1078-1093`, `:1244-1255`. |
| `dreams-thesis-wa/notebooks/cross_run_integration_master.ipynb` | notebook entry | Aggregates 12-condition matrix and produces cross-run tables/figures. | per-run `axis2_artifacts`, cross-axis join. | `dreams-thesis-wa/results/axis2/cross_run_integration/*.csv`, some figures. | `OUT_DIR` `:49`; writers `:282`, `:443`, `:636`, `:922`, `:1063`, `:1095`, `:1123`. |
| `dreams-thesis-wa/scripts/axis2_publication_figures.py` | script entry | Final Axis 2 publication figures/tables. | `model_runs`, `cross_axis_bridge`, cross-run artifacts. | `results/axis2/figures/*.png/pdf/csv/html/tex`. | data reads `:219-225`; figures `:413`, `:532`, `:583`, `:694`, `:712-713`, `:772`, `:936-937`; tables `:1002-1005`, `:1119-1120`, `:1151-1152`, `:1189`. |
| `dreams-thesis-wa/scripts/evaluate_holdout_maccs_bce.py` | script entry | Final holdout/R80 evaluation for MACCS-BCE only. | `model_runs/maccs_166_bce`, `holdout.parquet`, `full.hdf5`. | `holdout_metrics_maccs_bce.csv`, `retrieval/holdout_per_spectrum_ranks.csv`. | constants `:40-59`, checkpoint `:276`, outputs `:400-401`. |
| `dreams-thesis-wa/scripts/update_holdout_fixed_tau_maccs_bce.py` | script entry | Adds fixed OOD tau metrics to holdout metrics. | existing holdout metrics and optional prediction cache. | updates `holdout_metrics_maccs_bce.csv`. | `:24-36`, `:81`, `:162`. |
| `dreams-thesis-wa/legacy/notebooks/per_bit_morgan_analysis.ipynb` | legacy notebook | First Morgan/ECFP4 cosine run analysis; superseded. | first Morgan checkpoint/predictions. | deleted legacy `results/per_bit_analysis` and current `1st_morgan_2048_per_bit_analysis`. | output refs in `dreams-thesis-wa/legacy/notebooks/per_bit_morgan_analysis.ipynb:89`, `:1065`, `:2644`. |

### Axis 3 - deployment

| Path | Role | Purpose | Inputs | Outputs | Evidence |
|---|---|---|---|---|---|
| `dreams-thesis-wa/scripts/profile_axis3_mac_dda.py` | script entry | Profiles MAC-lab DDA raw data and determines label scope. | raw Axis 3 CSV/mzML dir. | Not committed: proprietary MAC profile, removed from public repo. | output constants `profile_axis3_mac_dda.py:33-35`, write at `:880`. |
| `dreams-thesis-wa/scripts/prepare_axis3_tier1_dataset.py` | script entry | Builds Tier 1 model-ready MAC dataset and libraries. | Axis 3 raw identity/fragments CSV, Axis 2 train/OOD. | `axis3_tier1_model_ready.hdf5`, libraries, seen flags, QC. | output constants `:47-61`, writers `:506`, `:576`, `:586`, `:699`, `:780-781`. |
| `dreams-thesis-wa/scripts/run_axis3_tier1_results.py` | script entry | Canonical Axis 3 retrieval/substructure-transfer run. | Axis 3 model-ready HDF5/libraries, MACCS-BCE fine-tuned and frozen Axis 2 runs, tau table. | `axis3_tier1_*`, `axis3_*` CSVs/PDFs/NPYs, `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` publication CSVs. | inputs `:51-54`, outputs `:56-89`, export constants `:68-70`, export writers `:837-859`, call site `:1285`. |
| `dreams-thesis-wa/scripts/axis3_publication_figures.py` | script consumer | Builds final Axis 3 thesis PDFs from specific Axis 3 CSVs. | `dreams-thesis-wa/results/axis3/specific/axis3_specific_*`. | `dreams-thesis-wa/results/axis3/figures/axis3_decomposition.pdf`, `axis3_substructure_transfer.pdf`. | constants `axis3_publication_figures.py:21-28`, saves `:201`, `:316`. |
| `dreams-thesis-wa/scripts/axis3_umap_figures.py` | dropped script | Builds UMAP embedding figures. Axis 3 UMAP/PCA visuals were dropped. | Axis 3 HDF5, `dreams-thesis-wa/results/axis3/specific/axis3_specific_per_spectrum_ranks.csv`, MACCS-BCE ckpt. | retired UMAP/PCA embeddings and PDFs, not in the canonical tracked layout. | constants `axis3_umap_figures.py:39-59`, saves `:173-174`, `:286`, `:327`. |
| `dreams-thesis-wa/legacy/scripts/partition_raw_mzml.py` | dropped Tier 2 helper | Partitions raw mzML files for Tier 2 recovery-from-raw path. | raw mzML dir. | analysis file list. | defaults `dreams-thesis-wa/legacy/scripts/partition_raw_mzml.py:11-18`, writer `:107`. |

### Cross-axis bridge

| Path | Role | Purpose | Inputs | Outputs | Evidence |
|---|---|---|---|---|---|
| `dreams-thesis-wa/scripts/cross_axis_bridge.py` | script entry | Joins Axis 1 descriptor probe results to Axis 2 per-bit AUROC; computes point-biserial equivalence and bridge figure. | `probe_indicator_merged.csv`, descriptor table, `model_runs`, `finetuning.hdf5`. | `dreams-thesis-wa/results/cross_axis/*.csv`, `.npy`, manifest, and `dreams-thesis-wa/results/cross_axis/figures/*.pdf`. | inputs `:153-159`, matrix writers `:468-469`, figure `:1012`, manifest `:1144`, tables `:1533-1582`. |
| `dreams-thesis-wa/notebooks/cross_axis_correlation_prep.ipynb` | older prep notebook | Computes descriptor-bit correlations for bridge prep. | descriptor matrix, fingerprints. | `results/cross_axis_correlation_prep/*`. | `OUT_DIR` `:74-76`, writers `:442-506`. |
| `dreams-thesis-wa/notebooks/cross_axis_bridge_analysis.ipynb` | notebook | Lightweight check/analysis around cross-axis outputs. | bridge CSVs. | no primary producer found. | only searched references found. |

### Shared/infra

| Path | Role | Purpose |
|---|---|---|
| `dreams-thesis-wa/scripts/aggregate_all_metrics.py` | helper | Aggregates model-run metrics to root and Axis 2 outputs; includes 12 run tags and seed 3407. |
| `dreams-thesis-wa/legacy/scripts/run_retrieval_evaluation.py` | legacy helper | Retrieval evaluation for old `results/per_bit_analysis`; hardcoded local checkpoint path. |
| `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` | retired historical record | Old one-off sign fixer; its `spearman_corr = -spearman_corr` convention is folded into `cross_axis_bridge.py`. |
| `dreams-thesis-wa/scripts/checks/verify_maccs.py`, `verify_splits.py`, `test_bce_loss.py`, `test_bce_quick*.{py,sh}` | checks | One-off validation/test utilities. |
| `dreams-thesis-wa/legacy/notebooks/debugging_local_results.ipynb` | scratch | Untracked local debugging notebook. |
| `dreams-thesis-wa/legacy/notebooks/axis_2_analysis_summary.ipynb` | legacy summary | Reads old `results/per_bit_analysis`; not final 12-condition matrix. |

## 3. Artifact provenance table

### Required result CSVs

| Artifact | Axis | Produced by | Input data | Output path | Evidence | Status |
|---|---|---|---|---|---|---|
| `probe_indicator_merged.csv` | Axis 1 | `task4_merge_probe_indicator.ipynb` | probe results + indicator tables | `dreams-thesis-wa/results/axis1/indicators/probe_indicator_merged.csv` | `task4_merge_probe_indicator.ipynb:503-504` | FULLY-TRACED |
| `bce_vs_cos_comparison_table.csv` | Axis 2 | `cross_run_integration_master.ipynb` | 12 per-run artifacts | `dreams-thesis-wa/results/axis2/cross_run_integration/bce_vs_cos_comparison_table.csv` | `cross_run_integration_master.ipynb:1062-1063` | FULLY-TRACED |
| `fine_tuned_vs_frozen_delta_table.csv` | Axis 2 | `cross_run_integration_master.ipynb` | 12 per-run artifacts | `dreams-thesis-wa/results/axis2/cross_run_integration/fine_tuned_vs_frozen_delta_table.csv` | `cross_run_integration_master.ipynb:921-922` | FULLY-TRACED |
| `fingerprint_family_comparison_table.csv` | Axis 2 | `cross_run_integration_master.ipynb` | 12 per-run artifacts | `dreams-thesis-wa/results/axis2/cross_run_integration/fingerprint_family_comparison_table.csv` | `cross_run_integration_master.ipynb:1094-1095` | FULLY-TRACED |
| `best_run_by_objective_table.csv` | Axis 2 | `cross_run_integration_master.ipynb` | 12 per-run artifacts | `dreams-thesis-wa/results/axis2/cross_run_integration/best_run_by_objective_table.csv` | `cross_run_integration_master.ipynb:1122-1123` | FULLY-TRACED |
| `axis2_master_comparison_table.csv` | Axis 2 | `axis2_publication_figures.py` | `model_runs` metrics | `results/axis2/figures/axis2_master_comparison_table.csv` | `axis2_publication_figures.py:1002-1004` | FULLY-TRACED |
| `axis2_optimal_tau_precision_recall_table.csv` | Axis 2 | `axis2_publication_figures.py` | threshold sweep tables | `results/axis2/figures/axis2_optimal_tau_precision_recall_table.csv` | `axis2_publication_figures.py:1151-1152` | FULLY-TRACED |
| per-run `retrieval_metrics.csv` | Axis 2 | `model_agnostic_eval_artifact_builder.ipynb` | per-run predictions/targets | `model_runs/<tag>/axis2_artifacts/retrieval/retrieval_metrics.csv` | `model_agnostic_eval_artifact_builder.ipynb:1092-1093` | FULLY-TRACED |
| `cross_run_summary_table.csv` | Axis 2 | `cross_run_integration_master.ipynb` | 12 per-run artifacts | `dreams-thesis-wa/results/axis2/cross_run_integration/cross_run_summary_table.csv` | `cross_run_integration_master.ipynb:282` | FULLY-TRACED |
| `fine_tuned_cross_axis_spearman_table.csv` | Cross-axis | `cross_axis_bridge.py` | Axis 1 + Axis 2 bridge tables | `dreams-thesis-wa/results/cross_axis/fine_tuned_cross_axis_spearman_table.csv` | producer `cross_axis_bridge.py:1577`; committed/tracked in Phase 5b; retired `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` convention folded in; scratch rerun verified value-identical to the committed table at atol 1e-6 | FULLY-TRACED |
| `maccs_top10_smarts_validation.csv` | Cross-axis | `cross_axis_bridge.py` | MACCS bit descriptors + SMARTS validation map | `dreams-thesis-wa/results/cross_axis/maccs_top10_smarts_validation.csv` | `cross_axis_bridge.py:1571` | FULLY-TRACED |
| `pointbiserial_equivalence_check.csv` | Cross-axis | `cross_axis_bridge.py` | descriptor/fingerprint matrix | `dreams-thesis-wa/results/cross_axis/pointbiserial_equivalence_check.csv` | `cross_axis_bridge.py:1568` | FULLY-TRACED |
| `axis3_specific_per_bit_auroc.csv` | Axis 3 | `run_axis3_tier1_results.py` | `axis3_perbit_auroc.csv` + cached fine-tuned predictions + HDF5 true bits/novel mask | `dreams-thesis-wa/results/axis3/specific/axis3_specific_per_bit_auroc.csv` | constants `run_axis3_tier1_results.py:68-70`, builder `:816-834`, writer `:854-859`; consumers `axis3_publication_figures.py:27`, `axis3_umap_figures.py:44` | FULLY-TRACED |
| `axis3_specific_retrieval_metrics.csv` | Axis 3 | `run_axis3_tier1_results.py` | `axis3_retrieval_metrics.csv` + ranks + compound recovery split by all/seen/novel | `dreams-thesis-wa/results/axis3/specific/axis3_specific_retrieval_metrics.csv` | builder `run_axis3_tier1_results.py:541-583`, writer `:846-849`; verified all-close after 2026-06-04 rerun | FULLY-TRACED |
| `axis3_specific_per_spectrum_ranks.csv` | Axis 3 | `run_axis3_tier1_results.py` | `axis3_tier1_per_spectrum_retrieval_results.csv` + ranks + seen flags | `dreams-thesis-wa/results/axis3/specific/axis3_specific_per_spectrum_ranks.csv` | builder `run_axis3_tier1_results.py:585-628`, writer `:850-852`; consumers `axis3_publication_figures.py:26`, `axis3_umap_figures.py:44` | FULLY-TRACED |
| `holdout_per_spectrum_ranks.csv` | Axis 2 holdout | `evaluate_holdout_maccs_bce.py` | holdout parquet, MACCS-BCE checkpoint | `model_runs/maccs_166_bce/axis2_artifacts/retrieval/holdout_per_spectrum_ranks.csv` | `evaluate_holdout_maccs_bce.py:58-59`, `:401` | FULLY-TRACED |
| `holdout_metrics_maccs_bce.csv` | Axis 2 holdout | `evaluate_holdout_maccs_bce.py`, then `update_holdout_fixed_tau_maccs_bce.py` | holdout predictions/metrics | `model_runs/maccs_166_bce/axis2_artifacts/holdout_metrics_maccs_bce.csv` | `evaluate_holdout_maccs_bce.py:400`; update `update_holdout_fixed_tau_maccs_bce.py:162` | AMBIGUOUS |

### Thesis figures and tables

| Artifact | Axis | Produced by | Output path on disk | Evidence | Status |
|---|---|---|---|---|---|
| Fig 1 workflow (schematic) | schematic | UNKNOWN - needs Wouter | UNKNOWN - needs Wouter | no producer found | ORPHAN |
| Fig 2 NPClassifier class distribution | Dataset | `create_superclass_heatmap.py` | `dreams-thesis-wa/results/shared/figures/superclass_heatmap.pdf` | `create_superclass_heatmap.py:91`, `:148` | FULLY-TRACED |
| Fig 3 DreaMS architecture (schematic) | schematic | UNKNOWN - needs Wouter | UNKNOWN - needs Wouter | no producer found | ORPHAN |
| Fig 4 SSL pre-training (schematic) | schematic | UNKNOWN - needs Wouter | UNKNOWN - needs Wouter | no producer found | ORPHAN |
| Fig 5 MAC curation (schematic) | schematic | UNKNOWN - needs Wouter | UNKNOWN - needs Wouter | no producer found | ORPHAN |
| Fig 6 R2 distributions linear vs MLP | Axis 1 | UNKNOWN - likely notebook outside searched producer set | `dreams-thesis-wa/results/axis1/figures/ssl_embedding_baseline_linear_vs_mlp*.{png,pdf}`, split panels | no direct producer found for final split panels | PARTIALLY-TRACED |
| Fig 7 indicator scatter | Axis 1 | `task5_indicator_vs_probe_figure.ipynb` | `dreams-thesis-wa/results/axis1/figures/indicator_vs_probe.{png,pdf}` or `scatter_standalone.{png,pdf}` | `task5_indicator_vs_probe_figure.ipynb:512-518`, `:778-783` | FULLY-TRACED |
| Fig 8 per-bit AUROC OOD distributions | Axis 2 | `axis2_publication_figures.py` | `results/axis2/figures/axis2_per_bit_auroc_distributions.{png,pdf}` | `axis2_publication_figures.py:327`, `:413`, `:1167-1169` | FULLY-TRACED |
| Fig 9 retrieval acc@k | Axis 2 | `axis2_publication_figures.py` | `results/axis2/figures/axis2_retrieval_accuracy_at_k.{png,pdf}` | `axis2_publication_figures.py:694` | FULLY-TRACED |
| Fig 10 fine-tuned-minus-frozen deltas | Axis 2 | `axis2_publication_figures.py` and older `cross_run_integration_master.ipynb` | `dreams-thesis-wa/results/axis2/figures/axis2_frozen_vs_finetuned_delta.{png,pdf}` and `dreams-thesis-wa/results/axis2/figures/fine_tuned_minus_frozen_heatmap.*` | script `axis2_publication_figures.py:879`, `:936-937`; notebook `cross_run_integration_master.ipynb:819-821` | AMBIGUOUS |
| Fig 11 cross-axis bridge | Cross-axis | `cross_axis_bridge.py` plus copied by `axis2_publication_figures.py` | `dreams-thesis-wa/results/cross_axis/figures/cross_axis_bridge_linear_r2_vs_auroc_ood.pdf`; `dreams-thesis-wa/results/axis2/figures/axis2_cross_axis_bridge_axis1_r2_vs_axis2_auroc.*` | `cross_axis_bridge.py:1012`; `axis2_publication_figures.py:700-713` | FULLY-TRACED |
| Fig 12 per-bit AUROC on MAC | Axis 3 | `run_axis3_tier1_results.py` | `dreams-thesis-wa/results/axis3/figures/axis3_tier1_substructure_transfer_scatter.pdf` | `run_axis3_tier1_results.py:986`, full Axis 3 version `:1084` | FULLY-TRACED |
| Fig 13 top-20 descriptors | Axis 1 appendix | UNKNOWN - needs Wouter | `dreams-thesis-wa/results/axis1/figures/top_20_descriptors.{png,pdf}` | output exists, no producer found | ORPHAN |
| Fig 14 bottom-20 descriptors | Axis 1 appendix | UNKNOWN - needs Wouter | `dreams-thesis-wa/results/axis1/figures/bottom_20_descriptors.pdf` and a retired pre-cleanup `bottom_20_descriptors.png` copy | output exists/deleted, no producer found | ORPHAN |
| Fig 15 val-vs-OOD generalisation gap | Axis 2 appendix | `axis2_publication_figures.py` | `results/axis2/figures/axis2_generalisation_gap.{png,pdf}` | `axis2_publication_figures.py:417`, `:532`, `:1163-1165` | FULLY-TRACED |
| Axis 3 decomposition final PDF | Axis 3 | Two possible producers | `dreams-thesis-wa/results/axis3/figures/axis3_tier1_decomposition_bars.pdf`, `dreams-thesis-wa/results/axis3/figures/axis3_decomposition.pdf` | `run_axis3_tier1_results.py:964`; full Axis 3 version `:1105`; `axis3_publication_figures.py:201` | AMBIGUOUS |
| Axis 3 UMAP PDFs | dropped | `axis3_umap_figures.py` | `dreams-thesis-wa/results/axis3/figures/axis3_umap_*.pdf` | `axis3_umap_figures.py:286`, `:327` | DROPPED |

Thesis tables T1-T14 are not consistently mapped to exact final LaTeX/table files in the repo. Local evidence:

- T1/T2 dataset/splits: source data and split methods are documented in `docs/thesis/THESIS_METHODS_EXTRACTION.md`; concrete split creation in `align_splits_for_finetuning.py`, but final thesis table file is UNKNOWN - needs Wouter.
- T3 probe comparison: likely Axis 1 probe CSVs/figures, but final table file is UNKNOWN - needs Wouter.
- T4 fingerprint families and T8 master 12-condition: `axis2_publication_figures.py` writes `axis2_master_comparison_table.csv/html` and `axis2_thesis_comparison_table.tex` at `:1002-1120`.
- T5 factorial design: represented by the 12 run-tag manifest below; no separate final table producer found.
- T6 hyperparameters: wrappers and configs give evidence, but no final table producer found.
- T7 descriptor families: `task3_descriptor_families.ipynb:401` writes `descriptor_family_summary.csv`.
- T9 Spearman bridge: `cross_axis_bridge.py:1577` writes `fine_tuned_cross_axis_spearman_table.csv`; retired `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` convention is folded into the bridge and Phase 5 verified the regenerated table value-identical to 1e-6.
- T10 Axis 3 retrieval/T14 full deployment metrics: `run_axis3_tier1_results.py:1227` writes Axis 3 retrieval metrics; `:846-859` writes `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` publication exports.
- T11 closed-pool top-1: contained in Axis 3 per-spectrum/retrieval tables; final table file UNKNOWN - needs Wouter.
- T12 MACCS bit-descriptor validation: `cross_axis_bridge.py:1571` writes `maccs_top10_smarts_validation.csv`.
- T13 Round-2 robustness: no final producer/table found. UNKNOWN - needs Wouter.

## 4. Provenance verification

### 4a. Run-tag manifest

All 12 cells of `{ECFP4/MACCS/MAP4} x {cosine/BCE} x {fine-tuned/frozen}` have a matching directory under `dreams-thesis-wa/results/model_runs`.

| Fingerprint | Loss | Backbone | Run tag | Path | Size | Checkpoint evidence | Status |
|---|---|---|---|---|---:|---|---|
| ECFP4/Morgan | cosine | fine-tuned | `morgan_2048_cos` | `model_runs/morgan_2048_cos` | 2.2G | `epoch=63-step=8382-val_loss=0.361643.ckpt`, 1.2G | OK |
| ECFP4/Morgan | BCE | fine-tuned | `morgan_2048_bce` | `model_runs/morgan_2048_bce` | 2.2G | `epoch=31-step=4224-val_loss=0.061447.ckpt`, 1.2G | OK |
| MACCS | cosine | fine-tuned | `maccs_166_cos` | `model_runs/maccs_166_cos` | 1.2G | `epoch=19-step=2640-val_loss=0.135409.ckpt`, 1.2G | OK |
| MACCS | BCE | fine-tuned | `maccs_166_bce` | `model_runs/maccs_166_bce` | 1.2G | `epoch=11-step=1584-val_loss=0.237082.ckpt`, 1.2G | OK; also holdout and Axis 3 source |
| MAP4 | cosine | fine-tuned | `map4_2048_cos` | `model_runs/map4_2048_cos` | 2.2G | `epoch=23-step=3168-val_loss=0.420152.ckpt`, 1.2G | OK |
| MAP4 | BCE | fine-tuned | `map4_2048_bce` | `model_runs/map4_2048_bce` | 2.2G | `epoch=17-step=2310-val_loss=0.453639.ckpt`, 1.2G | OK |
| ECFP4/Morgan | cosine | frozen | `morgan_2048_cos_frozen` | `model_runs/morgan_2048_cos_frozen` | 1.1G | `frozen_morgan_2048_cos_best.ckpt`, 36M plus alias `best.ckpt`, 36M | OK |
| ECFP4/Morgan | BCE | frozen | `morgan_2048_bce_frozen` | `model_runs/morgan_2048_bce_frozen` | 1.1G | `frozen_morgan_2048_bce_best.ckpt`, 36M plus alias `best.ckpt`, 36M | OK |
| MACCS | cosine | frozen | `maccs_166_cos_frozen` | `model_runs/maccs_166_cos_frozen` | 116M | `frozen_maccs_166_cos_best.ckpt`, 14M plus alias `best.ckpt`, 14M | OK |
| MACCS | BCE | frozen | `maccs_166_bce_frozen` | `model_runs/maccs_166_bce_frozen` | 116M | `frozen_maccs_166_bce_best.ckpt`, 14M plus alias `best.ckpt`, 14M | OK |
| MAP4 | cosine | frozen | `map4_2048_cos_frozen` | `model_runs/map4_2048_cos_frozen` | 1.1G | `frozen_map4_2048_cos_best.ckpt`, 36M plus alias `best.ckpt`, 36M | OK |
| MAP4 | BCE | frozen | `map4_2048_bce_frozen` | `model_runs/map4_2048_bce_frozen` | 1.1G | `frozen_map4_2048_bce_best.ckpt`, 36M plus alias `best.ckpt`, 36M | OK |

Flags:

- Missing cells: none.
- Duplicate matrix cells: none under `model_runs/`.
- Extra/superseded run: `results/1st_morgan_2048_per_bit_analysis/` is a legacy first ECFP4-cosine analysis, 1.1G, with old output paths `results/per_bit_analysis`. Current final Axis 2 scripts read `results/model_runs` (`axis2_publication_figures.py:94`, `:242-244`; `cross_run_integration_master.ipynb:49`, `:211`) and no final producer references `1st_morgan_2048_per_bit_analysis`. Therefore no reported thesis number should come from the first run instead of `morgan_2048_cos`, unless manually copied outside the scanned code.
- Frozen two-checkpoint pattern: each frozen `model_runs/*_frozen/checkpoints` has `frozen_<tag>_best.ckpt` plus `best.ckpt`. They are same byte size, but not symlinks or hardlinks: inode/link metadata shows different inodes and link count 1 for each. The canonical file is `frozen_<tag>_best.ckpt` because `run_config.json` points to it and `model_agnostic_eval_artifact_builder.ipynb:187-231` names those paths. Source `frozen_allpeaks_baselines.py:380-381` writes both separately. Byte hashes were not computed because checkpoint contents were not read.
- Current frozen H100 script tag mismatch: `h100_frozen_allpeaks_pipeline.sh:236-241` and `frozen_allpeaks_inference.py:51-56` use `_frozen_allpeaks` run tags, while current results use `frozen_<fp>_<loss>` in `frozen_deepsets_baselines/` and `<fp>_<loss>_frozen` in `model_runs/`. Current artifacts are therefore tied to the notebook/model-agnostic builder lineage, not fully to the latest wrapper defaults.

### 4b. Per-artifact provenance chains

Canonical chains:

1. Fine-tuned Axis 2: `fine_tune_test*.sh` or `fine_tune_round2.sh` -> `dreams/training/train.py` -> checkpoint in `results/finetuning`/checkpoint source -> `h100_batch_inference.py` or `model_agnostic_eval_artifact_builder.ipynb` -> `model_runs/<run>/axis2_artifacts/y_pred*.npy` -> `model_agnostic_eval_artifact_builder.ipynb` metrics -> `cross_run_integration_master.ipynb` and `axis2_publication_figures.py` tables/figures.
2. Frozen Axis 2: `frozen_embedding_deepsets_baselines.ipynb` -> `frozen_deepsets_baselines/frozen_*_best.ckpt` -> `model_agnostic_eval_artifact_builder.ipynb` -> `model_runs/<run>_frozen/axis2_artifacts/y_pred*.npy` -> same metric/figure chain as above.
3. Cross-axis bridge: `probe_indicator_merged.csv` + per-run AUROC -> `cross_axis_bridge.py` -> bridge CSVs/PDFs -> `axis2_publication_figures.py` copies/uses bridge figure/table.
4. Axis 3 canonical: `prepare_axis3_tier1_dataset.py` -> `axis3_tier1_model_ready.hdf5` and libraries -> `run_axis3_tier1_results.py` using `maccs_166_bce` and `maccs_166_bce_frozen` -> cached predictions -> Axis 3 metrics, per-spectrum ranks, transfer CSV, `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` publication exports, PDFs.
5. Axis 2 holdout/R80: `evaluate_holdout_maccs_bce.py` -> holdout metrics/ranks -> `update_holdout_fixed_tau_maccs_bce.py` modifies fixed-tau columns.

Prominent reproducibility holes:

- RESOLVED 2026-06-04: `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` are now emitted by `run_axis3_tier1_results.py`; rerun loaded cached fine-tuned/frozen predictions and regenerated the files. Verification: retrieval all-close for 12 rows, per-spectrum ranks equal for 6,236 rows with reciprocal ranks all-close, per-bit AUROC all-close for 166 bits.
- ORPHAN: schematic Figs 1, 3, 4, 5; no producer or source asset found.
- ORPHAN: `top_20_descriptors.*`, `bottom_20_descriptors.*`; outputs exist but producer not found.
- RESOLVED Phase 5b: `fine_tuned_cross_axis_spearman_table.csv` is committed/tracked and produced single-source by `cross_axis_bridge.py`; retired `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` remains historical only and the scratch rerun was value-identical to the committed table at atol 1e-6.
- AMBIGUOUS: `holdout_metrics_maccs_bce.csv` is produced by `evaluate_holdout_maccs_bce.py` and later updated by `update_holdout_fixed_tau_maccs_bce.py`.
- AMBIGUOUS: Axis 3 final PDFs exist under `dreams-thesis-wa/results/axis3/figures` with different producers/inputs.

### 4c. Config binding

| Run class | Saved config found | Thesis settings confirmation | Mismatches/gaps |
|---|---|---|---|
| Fine-tuned 6 runs | `model_runs/<run>/axis2_artifacts/run_config.json` only records run tag, checkpoint, fp kind, loss, sigmoid flag, output dir. No `hparams.yaml` found under `model_runs`. | Fingerprint/loss/checkpoint path are confirmed per run. Training wrappers show seed 3407, lr 1.5e-5, batch 256, max epochs 103 for original wrapper (`fine_tune_test.sh:137-145`) and fp64/scheduler Round 2 for BCE (`fine_tune_round2.sh:183-193`). | Local saved configs do not bind seed/lr/batch/backbone unfreeze to each fine-tuned checkpoint. Need WandB run config or checkpoint hparams. UNKNOWN - needs Wouter. |
| Frozen 6 runs | `frozen_deepsets_baselines/*/metadata.json` and `frozen_baseline_summary.csv`; notebook constants. | Notebook confirms seed 3407, lr 1.5e-5, batch 256 (`frozen_embedding_deepsets_baselines.ipynb:77-80`, `:250`, `:335`, `:442-445`). Metadata confirms lr/batch/max epochs/patience but not seed. | Current script defaults have batch-size 64 (`frozen_allpeaks_baselines.py:61`), but actual notebook results use 256. Current wrapper tags differ from current artifact tags. PARTIALLY-TRACED. |
| Axis 3 / holdout | no training config; uses MACCS-BCE run config/checkpoint. | Scripts hard-code `maccs_166_bce` (`run_axis3_tier1_results.py:51`; `evaluate_holdout_maccs_bce.py:40-44`). | Depends on fine-tuned MACCS-BCE config gap above. |

### 4d. Staleness and git state

| Producer | Git state | Producer mtime | Representative output mtime | Staleness flag |
|---|---|---:|---:|---|
| `dreams-thesis-wa/scripts/axis2_publication_figures.py` | UNTRACKED | 2026-05-27 09:46:59 | most Axis 2 figs 2026-03-31 14:44; retrieval fig 2026-05-27 09:47 | Producer modified after most outputs; verify current code reproduces March outputs. |
| `dreams-thesis-wa/notebooks/cross_run_integration_master.ipynb` | UNTRACKED | 2026-03-26 16:42:37 | outputs 2026-03-26 16:44 | OK for current outputs. |
| `dreams-thesis-wa/scripts/cross_axis_bridge.py` | UNTRACKED | 2026-05-05 10:13:36 | main bridge PDF 2026-03-31 12:22; Spearman table 2026-05-05 10:16 | Producer changed after bridge PDF; Spearman regenerated after script mtime. |
| `dreams-thesis-wa/scripts/run_axis3_tier1_results.py` | UNTRACKED | 2026-06-04 17:19:20 | Axis 3 outputs and `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` 2026-06-04 17:20 | OK; rerun loaded cached predictions, no GPU inference. |
| `dreams-thesis-wa/scripts/evaluate_holdout_maccs_bce.py` | UNTRACKED | 2026-06-03 11:01:35 | ranks 2026-06-03 11:07 | OK for initial holdout. |
| `dreams-thesis-wa/scripts/update_holdout_fixed_tau_maccs_bce.py` | UNTRACKED | 2026-06-03 13:07:41 | metrics 2026-06-03 13:13 | OK for fixed-tau update. |
| `dreams-thesis-wa/scripts/create_superclass_heatmap.py` | MODIFIED | 2026-05-26 22:23:17 | `superclass_heatmap.pdf` modified in git, output mtime not separately checked here | Producer modified; figure also modified. |
| `dreams-thesis-wa/notebooks/model_agnostic_eval_artifact_builder.ipynb` | UNTRACKED | 2026-03-26 14:25:18 | per-run artifacts mostly 2026-03-19 and 2026-03-26 | Some outputs predate notebook mtime; verify if current notebook reproduces earlier cached artifacts. |
| `dreams/training/train.py`, `dreams/models/heads/heads.py` | tracked-clean | 2026-03-30 14:54:00 | checkpoints 2026-03-16/17 | Producer changed after fine-tuned checkpoints; verify current code still reproduces checkpoints. |
| `dreams-thesis-wa/notebooks/frozen_embedding_deepsets_baselines.ipynb` | UNTRACKED | not in mtime sample; output 2026-03-19 | frozen outputs 2026-03-19 | UNKNOWN - needs mtime check if notebook remains canonical. |

Nothing is committed for the new thesis result set: many producers and outputs are UNTRACKED, while `create_superclass_heatmap.py`, `fine_tune_test.sh`, and `dreams-thesis-wa/legacy/notebooks/axis_2_analysis_summary.ipynb` are MODIFIED.

### 4e. Recompute recipes for headline numbers

These commands are recipes. Addendum exception: `python dreams-thesis-wa/scripts/run_axis3_tier1_results.py` was executed on 2026-06-04; it loaded cached fine-tuned/frozen prediction arrays and regenerated the `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` CSVs.

Axis 2 from archived predictions (no GPU):

```bash
# Requires archived `dreams-thesis-wa/results/model_runs_preds_only.zip` unpacked so that
# `dreams-thesis-wa/results/model_runs/<run_tag>/axis2_artifacts/y_pred*.npy`
# and `y_true*.npy` are present.
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/model_agnostic_eval_artifact_builder.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/cross_run_integration_master.ipynb
python dreams-thesis-wa/scripts/cross_axis_bridge.py
python dreams-thesis-wa/scripts/axis2_publication_figures.py
```

Axis 2 fine-tuned MACCS-BCE holdout:

```bash
# Requires archived checkpoint:
# dreams-thesis-wa/results/model_runs/maccs_166_bce/checkpoints/epoch=11-step=1584-val_loss=0.237082.ckpt
python dreams-thesis-wa/scripts/evaluate_holdout_maccs_bce.py
python dreams-thesis-wa/scripts/update_holdout_fixed_tau_maccs_bce.py
```

Axis 3 canonical deployment:

```bash
python dreams-thesis-wa/scripts/prepare_axis3_tier1_dataset.py
python dreams-thesis-wa/scripts/run_axis3_tier1_results.py
```

Axis 1 headline probing/indicator:

```bash
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/exploratory/probe_all_rdkit_descriptors.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/task0_indicator_setup.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/task1_build_knn_graphs.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/task2_indicator1_nn_consistency.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/task3_descriptor_families.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/task4_merge_probe_indicator.ipynb
jupyter nbconvert --to notebook --execute dreams-thesis-wa/notebooks/task5_indicator_vs_probe_figure.ipynb
```

## 5. Output and data inventory

### Large-file handling and archival manifest

Confirmed sizes:

- `dreams-thesis-wa/results/model_runs/`: 16G.
- `dreams-thesis-wa/results/model_runs_preds_only.zip`: 914M (`958133899` bytes).
- `dreams-thesis-wa/results/frozen_deepsets_baselines/`: 172M.

No file over 100 MB can enter git. The following exact `.gitignore` lines are recommended:

```gitignore
# Thesis large artifacts and raw data
model_runs/
dreams-thesis-wa/results/model_runs/
*.ckpt
*.pt
*.zip
frozen_deepsets_baselines/
dreams-thesis-wa/results/frozen_deepsets_baselines/
*.h5
*.hdf5
data/raw/
dreams-thesis-wa/data/raw/
**/data/raw/
```

The existing `.gitignore` already has broad `*ckpt`, `*pt`, `*.mzML`, `*.mzXML`, and `data/`, but it does not explicitly protect `model_runs/`, `*.zip`, or `frozen_deepsets_baselines/`.

| Recommendation | Run tag | Path | Size | Thesis result dependency | Evidence |
|---|---|---|---:|---|---|
| ARCHIVE TO ZENODO | all 12 Axis 2 runs | `dreams-thesis-wa/results/model_runs_preds_only.zip` | 914M / `958133899` bytes | All Axis 2 evaluation, cross-run tables, cross-axis bridge, publication figures without GPU | zip present; per-run prediction arrays present. |
| ARCHIVE TO ZENODO | `maccs_166_bce` | `dreams-thesis-wa/results/model_runs/maccs_166_bce/checkpoints/epoch=11-step=1584-val_loss=0.237082.ckpt` | 1.2G / `1243310996` bytes | Axis 3 deployment and Axis 2 holdout/R80 | `run_axis3_tier1_results.py:51`, `evaluate_holdout_maccs_bce.py:40-44`. |
| KEEP ON SNELLIUS | all 12 Axis 2 runs | `dreams-thesis-wa/results/model_runs/` | 16G | Full local replay with checkpoints plus predictions | H100 persistent output defaults to `$REPO_ROOT/dreams-thesis-wa/results/model_runs` (`h100_batch_inference.sh:35`, `h100_frozen_allpeaks_pipeline.sh:36`). Durable Snellius path beyond `$HOME/DreaMS/...` is UNKNOWN - needs Wouter. |
| GIT-IGNORE | frozen baselines | `dreams-thesis-wa/results/frozen_deepsets_baselines/` | 172M | Frozen baseline checkpoints and summary | checkpoint dirs contain 14M/36M ckpts. |
| COMMIT candidate | frozen baselines | `dreams-thesis-wa/results/frozen_deepsets_baselines/frozen_baseline_summary.csv` | 959B | frozen-half summary for the final 12-condition matrix | small CSV; contents list all six frozen runs. |
| COMMIT candidate | frozen baselines metadata | `dreams-thesis-wa/results/frozen_deepsets_baselines/*/{metadata.json,history.json}` | 429B-13K each | provenance for frozen baselines | small, but under ignored dir; consider copying to tracked manifest if needed. |

Per-run archive inventory:

| Run tag | Path | Size | Depends on / supports | Recommendation |
|---|---|---:|---|---|
| `maccs_166_cos` | `dreams-thesis-wa/results/model_runs/maccs_166_cos` | 1.2G | Axis 2 matrix, MACCS-cos fine-tuned | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `maccs_166_bce` | `dreams-thesis-wa/results/model_runs/maccs_166_bce` | 1.2G | Axis 2 matrix, Axis 3, holdout/R80 | ARCHIVE checkpoint + predictions; keep full on Snellius |
| `morgan_2048_cos` | `dreams-thesis-wa/results/model_runs/morgan_2048_cos` | 2.2G | Axis 2 ECFP4-cos fine-tuned | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `morgan_2048_bce` | `dreams-thesis-wa/results/model_runs/morgan_2048_bce` | 2.2G | Axis 2 ECFP4-BCE fine-tuned | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `map4_2048_cos` | `dreams-thesis-wa/results/model_runs/map4_2048_cos` | 2.2G | Axis 2 MAP4-cos fine-tuned | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `map4_2048_bce` | `dreams-thesis-wa/results/model_runs/map4_2048_bce` | 2.2G | Axis 2 MAP4-BCE fine-tuned | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `maccs_166_cos_frozen` | `dreams-thesis-wa/results/model_runs/maccs_166_cos_frozen` | 116M | Axis 2 MACCS-cos frozen | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `maccs_166_bce_frozen` | `dreams-thesis-wa/results/model_runs/maccs_166_bce_frozen` | 116M | Axis 2 MACCS-BCE frozen; Axis 3 comparison | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `morgan_2048_cos_frozen` | `dreams-thesis-wa/results/model_runs/morgan_2048_cos_frozen` | 1.1G | Axis 2 ECFP4-cos frozen | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `morgan_2048_bce_frozen` | `dreams-thesis-wa/results/model_runs/morgan_2048_bce_frozen` | 1.1G | Axis 2 ECFP4-BCE frozen | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `map4_2048_cos_frozen` | `dreams-thesis-wa/results/model_runs/map4_2048_cos_frozen` | 1.1G | Axis 2 MAP4-cos frozen | IGNORE+ARCHIVE predictions; keep full on Snellius |
| `map4_2048_bce_frozen` | `dreams-thesis-wa/results/model_runs/map4_2048_bce_frozen` | 1.1G | Axis 2 MAP4-BCE frozen | IGNORE+ARCHIVE predictions; keep full on Snellius |

Other output/data recommendations:

| Path | Size | Recommendation | Note |
|---|---:|---|---|
| `dreams-thesis-wa/results/axis2/cross_run_integration/` | small to 2.0M files | COMMIT | canonical Axis 2 summary CSVs/figures. |
| `dreams-thesis-wa/results/cross_axis/` | small to 4.9M files | COMMIT | canonical bridge outputs; exclude if binary policy wants regenerated figures only. |
| `dreams-thesis-wa/results/axis2/figures/` | small | COMMIT | final Axis 2 figures/tables. |
| `dreams-thesis-wa/results/axis3/` | mostly small, one 6.1M embedding NPY | COMMIT CSV/PDF/JSON; IGNORE NPY/HDF5 if raw/regenerable | canonical Axis 3 outputs. |
| retired pre-cleanup Axis 3 result copies | 13M | RETIRED | publication exports now live under `dreams-thesis-wa/results/axis3/specific/`; dropped UMAP files stay out of the final tracked layout. |
| `dreams-thesis-wa/results/axis3/figures/` | 216K | UNCERTAIN | final export folder, but includes dropped UMAP PDFs. |
| `dreams-thesis-wa/results/1st_morgan_2048_per_bit_analysis/` | 1.1G | LIKELY-DELETE or ARCHIVE AS LEGACY | superseded by `model_runs/morgan_2048_cos`; contains large NPY/PT arrays. |
| deleted `dreams-thesis-wa/results/per_bit_analysis/**` | n/a | LIKELY-DELETE | legacy old path deleted in git status. |
| `dreams-thesis-wa/data/raw/`, `dreams-thesis-wa/data/processed/` | unknown | IGNORE | raw/processed data should not enter git. |

## 6. Dead / superseded / duplicate candidates

| Candidate | Why it looks obsolete | Superseded by | Action |
|---|---|---|---|
| `dreams-thesis-wa/results/1st_morgan_2048_per_bit_analysis/` | First ECFP4-cos run, 1.1G, old path. Not read by final 12-condition scripts. | `model_runs/morgan_2048_cos` | Flag only. |
| `dreams-thesis-wa/legacy/notebooks/per_bit_morgan_analysis.ipynb` | Writes old `results/per_bit_analysis`; final model-agnostic builder handles all 12. | `model_agnostic_eval_artifact_builder.ipynb` | Flag only. |
| deleted `results/per_bit_analysis` figures | Deleted in git status; old Morgan-only outputs. | `model_runs/<tag>/axis2_artifacts` and `results/axis2/figures` | Flag only. |
| `dreams-thesis-wa/legacy/scripts/run_retrieval_evaluation.py` | Hardcoded old local checkpoint and old output dir. | model-agnostic builder per-run retrieval | Flag only. |
| `dreams-thesis-wa/legacy/notebooks/axis_2_analysis_summary.ipynb` | Reads old `results/per_bit_analysis`; modified. | cross-run integration + publication script | Flag only. |
| `axis3_umap_figures.py` and `dreams-thesis-wa/results/axis3/figures/axis3_umap_*.pdf` | UMAP/PCA visuals were dropped. | no replacement; omitted thesis visuals | Flag only. |
| `dreams-thesis-wa/legacy/scripts/partition_raw_mzml.py` | Tier 2 recovery-from-raw path dropped. | Tier 1 prepared dataset path | Flag only. |
| `dreams-thesis-wa/legacy/notebooks/debugging_local_results.ipynb` | untracked scratch/debugging notebook. | no canonical role found | Flag only. |
| `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` | retired one-off sign fixer. | `cross_axis_bridge.py` with folded sign convention | Historical record only. |
| `dreams-thesis-wa/results/axis3/specific/axis3_specific_*` | previously manual/implicit export. | explicit `run_axis3_tier1_results.py` publication export step. | Resolved 2026-06-04. |

## 7. Reproduction pipeline

Axis 1:

```text
MassSpecGym spectra/data
  -> generate_ssl_embeddings.py / probe notebooks
  -> all_descriptors_probing_results*.csv/pkl
  -> task0-task4 indicator notebooks
  -> probe_indicator_merged.csv
  -> task5_indicator_vs_probe_figure.ipynb
  -> Fig 7 and Axis 1 tables
```

Axis 2:

```text
MassSpecGym split HDF5 + pretrained ssl_model.ckpt
  -> fine_tune_test*.sh / fine_tune_round2.sh -> dreams/training/train.py
  -> fine-tuned checkpoints
  -> h100_batch_inference.py or model_agnostic_eval_artifact_builder.ipynb
  -> model_runs/<fine-run>/axis2_artifacts/y_pred*.npy, y_true*.npy

finetuning_with_ssl_embeddings.hdf5 + fingerprint_cache.npz
  -> frozen_embedding_deepsets_baselines.ipynb
  -> frozen_deepsets_baselines/frozen_*_best.ckpt
  -> model_agnostic_eval_artifact_builder.ipynb
  -> model_runs/<frozen-run>/axis2_artifacts/y_pred*.npy, y_true*.npy

all 12 model_runs axis2_artifacts
  -> model_agnostic_eval_artifact_builder.ipynb
  -> per-run AUROC/threshold/retrieval CSVs
  -> cross_run_integration_master.ipynb
  -> axis2_publication_figures.py
  -> final Axis 2 tables/figures
```

Cross-axis:

```text
Axis 1 probe_indicator_merged.csv + descriptor table + Axis 2 per-bit AUROC
  -> cross_axis_bridge.py
  -> point-biserial matrices, Spearman table, SMARTS validation, bridge figure
```

Axis 3:

```text
MAC raw identity/fragments + Axis 2 train/OOD references
  -> profile_axis3_mac_dda.py
  -> prepare_axis3_tier1_dataset.py
  -> axis3_tier1_model_ready.hdf5 + closed/open libraries
  -> run_axis3_tier1_results.py with maccs_166_bce and maccs_166_bce_frozen
  -> Axis 3 retrieval, per-spectrum ranks, substructure transfer, specific Axis 3 exports, figures
```

Holdout/R80:

```text
holdout.parquet + full.hdf5 + maccs_166_bce checkpoint
  -> evaluate_holdout_maccs_bce.py
  -> update_holdout_fixed_tau_maccs_bce.py
  -> holdout metrics and per-spectrum ranks
```

## 8. Environment and dependencies

Python and package evidence:

- `setup.py` declares `python_requires='>=3.11'` and pins core dependencies including `numpy==1.25.0`, `torch==2.2.1`, `pytorch-lightning==2.0.8`, `pandas==2.2.1`, `pyarrow==15.0.2`, `h5py==3.11.0`, `rdkit==2023.9.6`, `umap-learn==0.5.6`, `seaborn==0.13.2`, `wandb==0.16.4`.
- `dreams-thesis-wa/requirements.txt` lists `pandas`, `numpy`, `scipy`, `scikit-learn`, `rdkit`, `map4`, `matplotlib`, `seaborn`, `umap-learn`, `adjustText`, `jupyter`, `ipykernel`, `uv`.
- `model_agnostic_eval_artifact_builder.ipynb` output mentions `.venv/lib/python3.10/...`, which conflicts with `setup.py` Python `>=3.11`. UNKNOWN - needs Wouter which environment produced final artifacts.

Hardcoded paths that can break reproduction:

- `dreams-thesis-wa/legacy/scripts/run_retrieval_evaluation.py:26` local project root `/Users/wouterachterberg/coding/DreaMS`.
- `dreams-thesis-wa/legacy/scripts/run_retrieval_evaluation.py:91` local external checkpoint `/Volumes/NVMe_Wouter/THESIS/snellius_output/...`.
- `dreams-thesis-wa/scripts/_inspect_ckpt.py:6` local external checkpoint `/Volumes/NVMe_Wouter/...`.
- `dreams-thesis-wa/scripts/h100_batch_inference.py:90` default checkpoint base `/Volumes/NVMe_Wouter/THESIS/model_checkpoints`.
- `dreams-thesis-wa/notebooks/model_agnostic_eval_artifact_builder.ipynb:133` same `/Volumes/NVMe_Wouter/THESIS/model_checkpoints`.
- `dreams-thesis-wa/scripts/prepare_axis3_tier1_dataset.py:23`, `profile_axis3_mac_dda.py:25`, `run_axis3_tier1_results.py:21` hardcode `/Users/wouterachterberg/coding/DreaMS`.
- `dreams-thesis-wa/scripts/prepare_axis3_tier1_dataset.py:42`, `profile_axis3_mac_dda.py:26` hardcode `data/raw/axis_3_data`.
- `dreams-thesis-wa/scripts/fine_tune_test.sh:41`, `fine_tune_round2.sh:120`, `h100_batch_inference.sh:44`, `h100_frozen_allpeaks_pipeline.sh:122` hardcode Snellius `/scratch-shared/$USER/...` scratch patterns.
- `dreams-thesis-wa/scripts/fine_tune_test.sh:47`, `:59`, `fine_tune_round2.sh:126`, `:134`, `h100_batch_inference.sh:35`, `:72-73` assume `$HOME/DreaMS`.
- `dreams-thesis-wa/legacy/notebooks/DreaMS.code-workspace:7`, `:10` references relative paths to `/Volumes/NVMe_Wouter/THESIS`.
- upstream/demo scripts under root `scripts/` include `/Users/maxvandenboom/...` and mzML demo paths; likely not thesis reproduction paths but still nonportable.
- result manifests also contain absolute local paths, e.g. `dreams-thesis-wa/results/cross_axis/analysis_manifest.json`, `dreams-thesis-wa/results/axis3/*_summary.json`, and per-run `run_config.json`.

## 9. Open questions for Wouter

1. UNKNOWN - needs Wouter: Where is the durable Snellius path for the full 16G `model_runs/` set? Code documents `$HOME/DreaMS/.../results/model_runs` and scratch paths, but not a stable archive path.
2. UNKNOWN - needs Wouter: Which exact WandB run configs/hparams bind each fine-tuned checkpoint to seed 3407, lr 1.5e-5, batch, precision, loss, and backbone mode?
3. RESOLVED Phase 6d: canonical thesis figures now live under `dreams-thesis-wa/results/{axis1,axis2,axis3,cross_axis,shared}/figures` as appropriate.
4. UNKNOWN - needs Wouter: What are the source files for schematic Figs 1, 3, 4, and 5?
5. UNKNOWN - needs Wouter: What produces `top_20_descriptors.*`, `bottom_20_descriptors.*`, and the final split-panel Axis 1 R2 figures?
7. RESOLVED Phase 5: `dreams-thesis-wa/legacy/scripts/fix_spearman_sign.py` is retired; `cross_axis_bridge.py` produces the Spearman bridge table single-source.
8. UNKNOWN - needs Wouter: Should small frozen JSON histories/metadata be committed as provenance, copied out of the ignored frozen checkpoint directory, or left for Zenodo only?
9. UNKNOWN - needs Wouter: Which Python environment produced final artifacts, given `setup.py` says Python >=3.11 but notebook output references Python 3.10?
