#!/usr/bin/env python3
"""
Orchestrate frozen all-peaks baselines for all 6 conditions.

Steps:
1) Optionally create ssl_peak_embeddings in HDF5 (one-time extraction)
2) Train frozen all-peaks DeepSets heads for six fp/loss conditions
3) Print MODEL_SPECS_ADDITION entries to paste into model_agnostic_eval_artifact_builder

Usage:
    python dreams-thesis-wa/src/run_frozen_allpeaks.py --run-all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RUNS = [
    ("morgan_2048", "bce_logits", "morgan_2048_bce_frozen_allpeaks"),
    ("maccs_166", "bce_logits", "maccs_166_bce_frozen_allpeaks"),
    ("map4_2048", "bce_logits", "map4_2048_bce_frozen_allpeaks"),
    ("morgan_2048", "cos", "morgan_2048_cos_frozen_allpeaks"),
    ("maccs_166", "cos", "maccs_166_cos_frozen_allpeaks"),
    ("map4_2048", "cos", "map4_2048_cos_frozen_allpeaks"),
]


def parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Run frozen all-peaks baselines.")
    p.add_argument("--project-root", type=Path, default=root)
    p.add_argument("--extract-peaks", action="store_true", help="Run peak extraction before training.")
    p.add_argument("--extract-overwrite", action="store_true", help="Overwrite existing ssl_peak_embeddings.")
    p.add_argument("--extract-dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--extract-folds", type=str, default="all", help="all or comma-separated folds (e.g. train,val)")
    p.add_argument("--run-all", action="store_true", help="Train all six conditions.")
    p.add_argument("--run-tag", type=str, default=None, help="Train one specific run tag from preset list.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=103)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--use-hdf5-peak-mask", action="store_true")
    return p.parse_args()


def run_cmd(cmd):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def maybe_extract(args, py_exe, src_dir):
    if not args.extract_peaks:
        return

    cmd = [
        py_exe,
        src_dir / "create_peak_embeddings.py",
        "--dtype",
        args.extract_dtype,
        "--folds",
        args.extract_folds,
    ]
    if args.extract_overwrite:
        cmd.append("--overwrite")
    cmd.extend(["--write-peak-mask"])
    run_cmd(cmd)


def selected_runs(args):
    if args.run_all:
        return RUNS
    if args.run_tag:
        out = [r for r in RUNS if r[2] == args.run_tag]
        if not out:
            known = ", ".join(r[2] for r in RUNS)
            raise ValueError(f"Unknown --run-tag {args.run_tag}. Known tags: {known}")
        return out
    raise ValueError("Provide either --run-all or --run-tag")


def print_model_specs_addition(project_root: Path):
    print("\nMODEL_SPECS_ADDITION = [")
    for fp_kind, loss_kind, run_tag in RUNS:
        ckpt = project_root / "dreams-thesis-wa/results/frozen_allpeaks_baselines" / run_tag / "checkpoints" / f"{run_tag}_best.ckpt"
        apply_sigmoid = "True" if loss_kind == "bce_logits" else "False"
        loss_short = "bce" if loss_kind == "bce_logits" else "cos"
        print("    {")
        print(f"        'run_tag': '{run_tag}',")
        print(f"        'fp_kind': '{fp_kind}',")
        print(f"        'loss_kind': '{loss_short}',")
        print("        'model_type': 'frozen_head_allpeaks',")
        print(f"        'ckpt_path': str(PROJECT_ROOT / '{ckpt.relative_to(project_root).as_posix()}'),")
        print(f"        'apply_sigmoid_to_pred': {apply_sigmoid},")
        print("    },")
    print("]")

    print("\nNote:")
    print("- model_agnostic_eval_artifact_builder.ipynb needs a 'frozen_head_allpeaks' branch")
    print("  that loads ssl_peak_embeddings [N,P,1024] and runs model(x, mask).")


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    src_dir = project_root / "dreams-thesis-wa/src"
    py_exe = Path(sys.executable)

    maybe_extract(args, py_exe, src_dir)

    runs = selected_runs(args)
    for fp_kind, loss_kind, run_tag in runs:
        cmd = [
            py_exe,
            src_dir / "frozen_allpeaks_baselines.py",
            "--project-root",
            project_root,
            "--fp-kind",
            fp_kind,
            "--loss-kind",
            loss_kind,
            "--run-tag",
            run_tag,
            "--batch-size",
            str(args.batch_size),
            "--max-epochs",
            str(args.max_epochs),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.device is not None:
            cmd.extend(["--device", args.device])
        if args.use_hdf5_peak_mask:
            cmd.append("--use-hdf5-peak-mask")

        run_cmd(cmd)

    print_model_specs_addition(project_root)


if __name__ == "__main__":
    main()
