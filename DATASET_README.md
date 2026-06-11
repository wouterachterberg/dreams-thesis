# DreaMS thesis - data and model archive

Accompanies the repository https://github.com/wouterachterberg/dreams-thesis
(release v1.2-clean) and the thesis "From Spectra to Structure"
(W. Achterberg, VU Amsterdam, 2026). Cite via DOI 10.5281/zenodo.20598853.

## Contents
| File (Zenodo) | Size | Restore to (repo path) | Purpose |
|---|---|---|---|
| model_runs_preds_only.zip | ~4.3 GB | dreams-thesis-wa/results/model_runs/ | Predicted fingerprints for all 12 conditions (Tier 1 input) |
| maccs_bce_finetuned.ckpt | ~1.2 GB | dreams-thesis-wa/results/model_runs/maccs_166_bce/checkpoints/epoch=11-step=1584-val_loss=0.237082.ckpt | Fine-tuned MACCS-BCE model - the reported model, used for Axis 3 and the holdout (Tier 2 input) |
| maccs_bce_frozen.ckpt (optional) | ~15 MB | dreams-thesis-wa/results/model_runs/maccs_166_bce_frozen/checkpoints/ (keep original filename) | Frozen-backbone MACCS-BCE baseline, for the frozen-vs-fine-tuned comparison |
| all_rdkit_descriptors.parquet | 17 MB (17,388,387 bytes) | dreams-thesis-wa/data/processed/massspecgym_complete/all_rdkit_descriptors.parquet | 201 RDKit descriptors per MassSpecGym molecule (Axis 1 / cross-axis input) |

When restoring the checkpoints, keep the original filenames shown above; the
evaluation and deployment scripts locate them by name.

The four-way Murcko split manifest (splits.csv) is not in this archive - it ships
in the repository at dreams-thesis-wa/results/shared/splits.csv and so travels
with the code DOI.

## Not included (fetch from source)
- MassSpecGym spectra and structures: huggingface.co/datasets/roman-bushuiev/MassSpecGym (CC BY 4.0), pinned revision 1b6d1ec69122aaa35694f531c35f0ea6a01bec52.
- DreaMS pre-trained backbone: pluskal-lab original distribution.
- MAC DDA dataset (Axis 3 input): controlled-access (see thesis Data and Code Availability). The aggregate Axis 3 outputs that back the reported results are tracked in the repository under dreams-thesis-wa/results/axis3/; per-spectrum compound identities are withheld because the MAC compound library is proprietary.

## Reproduce the splits
Run the Murcko-scaffold split script (seed 3407) against the pinned MassSpecGym
revision, or join splits.csv (fold column) onto MassSpecGym by spectrum id. Both
give identical folds.

## Load the checkpoint
Restore maccs_bce_finetuned.ckpt to the path above. The holdout evaluator selects
the best validation-loss checkpoint from
`dreams-thesis-wa/results/model_runs/maccs_166_bce/checkpoints/` with
`resolve_best_checkpoint(CHECKPOINT_DIR)` and then loads it through
`load_model_compat(best_ckpt, device)` from `dreams-thesis-wa/scripts/h100_batch_inference.py`.
Axis 3 uses the same compatibility loader after resolving the fine-tuned
checkpoint with `best_checkpoint_from_dir(FINE_TUNED_RUN_DIR, allow_best_name=False)`.

## Reproduction tiers
- Tier 0: repository only - read the tracked tables and figures.
- Tier 1: unpack this archive into dreams-thesis-wa/results/ and dreams-thesis-wa/data/, then run the producers (CPU, minutes).
- Tier 2: re-run inference from maccs_bce_finetuned.ckpt (GPU).
- Tier 3: full retrain (MassSpecGym + DreaMS backbone + cluster compute).
