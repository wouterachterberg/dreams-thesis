#!/usr/bin/env python3
"""Run batch inference for Axis 2 artifacts on GPU clusters (e.g., H100).

This script only computes and saves model predictions:
- axis2_artifacts/y_pred.npy      (OOD)
- axis2_artifacts/y_pred_val.npy  (validation)

Ground-truth fingerprint arrays (y_true*) can be computed locally once per fp kind.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import pathlib
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Allow imports from repository root when run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_CHECKPOINTS_DIR
from dreams.definitions import FOLD, PRETRAINED
from dreams.models.heads.heads import FingerprintHead
from dreams.utils import data as du
from dreams.utils.dformats import DataFormatBuilder

DEFAULT_MODEL_SPECS = [
    {
        "run_tag": "morgan_2048_cos",
        "fp_kind": "morgan_2048",
        "loss_kind": "cos",
        "ckpt_file": "epoch=63-step=8382-val_loss=0.361643.ckpt",
        "apply_sigmoid_to_pred": False,
    },
    {
        "run_tag": "morgan_2048_bce",
        "fp_kind": "morgan_2048",
        "loss_kind": "bce",
        "ckpt_file": "epoch=31-step=4224-val_loss=0.061447.ckpt",
        "apply_sigmoid_to_pred": True,
    },
    {
        "run_tag": "maccs_166_cos",
        "fp_kind": "maccs_166",
        "loss_kind": "cos",
        "ckpt_file": "epoch=19-step=2640-val_loss=0.135409.ckpt",
        "apply_sigmoid_to_pred": False,
    },
    {
        "run_tag": "maccs_166_bce",
        "fp_kind": "maccs_166",
        "loss_kind": "bce",
        "ckpt_file": "epoch=11-step=1584-val_loss=0.237082.ckpt",
        "apply_sigmoid_to_pred": True,
    },
    {
        "run_tag": "map4_2048_cos",
        "fp_kind": "map4_2048",
        "loss_kind": "cos",
        "ckpt_file": "epoch=23-step=3168-val_loss=0.420152.ckpt",
        "apply_sigmoid_to_pred": False,
    },
    {
        "run_tag": "map4_2048_bce",
        "fp_kind": "map4_2048",
        "loss_kind": "bce",
        "ckpt_file": "epoch=17-step=2310-val_loss=0.453639.ckpt",
        "apply_sigmoid_to_pred": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for DreaMS Axis 2 runs.")
    parser.add_argument(
        "--ckpt-base-dir",
        type=Path,
        default=MODEL_CHECKPOINTS_DIR,
        help="Directory containing checkpoint files from specs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/results/model_runs",
        help="Root folder for per-run outputs.",
    )
    parser.add_argument(
        "--probing-test",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet",
        help="Path to probing_test parquet file.",
    )
    parser.add_argument(
        "--finetuning-hdf5",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5",
        help="Path to finetuning.hdf5.",
    )
    parser.add_argument(
        "--specs-json",
        type=Path,
        default=None,
        help="Optional JSON file containing model specs list.",
    )
    parser.add_argument(
        "--run-tags",
        type=str,
        default=None,
        help="Comma-separated run tags to execute. Default: all specs.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="none",
        choices=["none", "float16", "bfloat16"],
        help=(
            "Optional CUDA autocast dtype. Defaults to disabled because mixed precision "
            "changed fingerprint head predictions materially in validation checks."
        ),
    )
    parser.add_argument(
        "--regression-check-batch-size",
        type=int,
        default=32,
        help=(
            "Number of validation samples to compare between the training dataloader path "
            "and the script preprocessing path before full inference."
        ),
    )
    parser.add_argument(
        "--skip-regression-check",
        action="store_true",
        help="Skip the validation regression check that compares the first batch against the training pipeline.",
    )
    return parser.parse_args()


def parse_spectrum_strings(mzs_str: str, intens_str: str, n_peaks: int = 128) -> np.ndarray:
    mzs = np.fromstring(str(mzs_str), sep=",", dtype=np.float32)
    ints = np.fromstring(str(intens_str), sep=",", dtype=np.float32)
    if len(mzs) == 0 or len(ints) == 0:
        return np.zeros((2, n_peaks), dtype=np.float32)

    n = min(len(mzs), len(ints))
    mzs, ints = mzs[:n], ints[:n]
    order = np.argsort(ints)[::-1][:n_peaks]
    mzs, ints = mzs[order], ints[order]
    order_mz = np.argsort(mzs)
    mzs, ints = mzs[order_mz], ints[order_mz]
    if ints.max() > 0:
        ints = ints / ints.max()

    out = np.zeros((2, n_peaks), dtype=np.float32)
    out[0, : len(mzs)] = mzs
    out[1, : len(ints)] = ints
    return out


def ensure_spectrum_layout(spec: np.ndarray) -> np.ndarray:
    """Convert spectra to shape [N, 128, 2] expected by the model."""
    arr = np.asarray(spec, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D spectrum array, got shape={arr.shape}")

    if arr.shape[-1] == 2:
        return arr

    if arr.shape[1] == 2:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)

    raise ValueError(
        f"Unsupported spectrum layout {arr.shape}; expected [N, 128, 2] or [N, 2, 128]."
    )


def load_datasets(probing_test: Path, finetuning_hdf5: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading OOD data from: {probing_test}")
    df_ood = pd.read_parquet(probing_test)
    spec_ood = np.stack(
        [parse_spectrum_strings(m, i) for m, i in zip(df_ood["mzs"], df_ood["intensities"])],
        axis=0,
    )
    spec_ood = ensure_spectrum_layout(spec_ood)
    
    # Extract precursor_mz for OOD
    prec_ood = np.full(len(df_ood), np.nan, dtype=np.float32)
    for prec_col in ["precursor_mz", "PRECURSOR_MZ", "prec_mz", "PRECURSOR M/Z"]:
        if prec_col in df_ood.columns:
            prec_ood = df_ood[prec_col].values.astype(np.float32)
            break

    print(f"Loading VAL data from: {finetuning_hdf5}")
    with h5py.File(finetuning_hdf5, "r") as f:
        fold = np.array([
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f["fold"][:]
        ])
        spec_all = f["spectrum"][:].astype(np.float32)
        
        # Extract precursor_mz for VAL
        prec_all = np.full(len(spec_all), np.nan, dtype=np.float32)
        if "precursor_mz" in f:
            prec_all = f["precursor_mz"][:].astype(np.float32)

    val_mask = fold == "val"
    if val_mask.sum() == 0:
        val_mask = fold == "test"

    spec_val = spec_all[val_mask]
    spec_val = ensure_spectrum_layout(spec_val)
    prec_val = prec_all[val_mask]

    print(f"OOD spectra: {spec_ood.shape}, precursor_mz: {prec_ood.shape}")
    print(f"VAL spectra: {spec_val.shape}, precursor_mz: {prec_val.shape}")
    return spec_ood, spec_val, prec_ood, prec_val


def load_specs(specs_json: Path | None) -> list[dict]:
    if specs_json is None:
        return DEFAULT_MODEL_SPECS

    specs = json.loads(specs_json.read_text())
    if not isinstance(specs, list):
        raise ValueError("specs-json must contain a JSON list.")

    required = {"run_tag", "fp_kind", "loss_kind", "apply_sigmoid_to_pred"}
    for s in specs:
        missing = required - set(s.keys())
        if missing:
            raise ValueError(f"Spec missing keys {missing}: {s}")
    return specs


def parse_val_loss_from_name(path: Path) -> float | None:
    m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", path.name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def discover_checkpoint_for_spec(ckpt_base_dir: Path, spec: dict) -> Path:
    """Find best checkpoint for a spec by searching recursively in nested run folders."""
    ckpt_file = spec.get("ckpt_file")
    if ckpt_file:
        exact = [p for p in ckpt_base_dir.rglob(ckpt_file) if p.is_file()]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            with_loss = [(parse_val_loss_from_name(p), p) for p in exact]
            with_loss = [(vl, p) for vl, p in with_loss if vl is not None]
            if with_loss:
                with_loss.sort(key=lambda x: x[0])
                return with_loss[0][1]
            exact.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return exact[0]

    fp_objective = f"fp_{spec['fp_kind']}".lower()
    loss_token = "bce_logits" if spec["loss_kind"] == "bce" else "cos"

    all_ckpts = [p for p in ckpt_base_dir.rglob("*.ckpt") if p.is_file()]
    if not all_ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_base_dir}")

    primary = []
    fallback = []
    for p in all_ckpts:
        full = str(p).lower()
        if fp_objective in full and loss_token in full:
            primary.append(p)
        elif fp_objective in full:
            fallback.append(p)

    candidates = primary if primary else fallback
    if not candidates:
        raise FileNotFoundError(
            f"Could not find checkpoint for run_tag={spec['run_tag']} "
            f"(expected tokens: {fp_objective}, {loss_token}) under {ckpt_base_dir}."
        )

    with_loss = [(parse_val_loss_from_name(p), p) for p in candidates]
    with_loss = [(vl, p) for vl, p in with_loss if vl is not None]

    if with_loss:
        # Pick smallest validation loss among matching checkpoints.
        with_loss.sort(key=lambda x: x[0])
        return with_loss[0][1]

    # Fallback: prefer non-last checkpoints, then latest mtime.
    non_last = [p for p in candidates if p.name != "last.ckpt"]
    pool = non_last if non_last else candidates
    pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pool[0]


def load_model_compat(ckpt_path: Path, device: str) -> FingerprintHead:
    """Load PL checkpoint with compatibility for torch>=2.6 weights_only behavior."""
    try:
        torch.serialization.add_safe_globals([pathlib.PosixPath])
    except Exception:
        pass

    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat
    try:
        model = FingerprintHead.load_from_checkpoint(
            str(ckpt_path),
            backbone=PRETRAINED / "ssl_model.ckpt",
            map_location=torch.device("cpu"),
        )
    finally:
        torch.load = original_torch_load

    return model.eval().float().to(device)


def build_train_like_spec_preprocessor() -> du.SpectrumPreprocessor:
    return du.SpectrumPreprocessor(
        dformat=DataFormatBuilder("A").get_dformat(),
        prec_intens=1.1,
        n_highest_peaks=100,
        spec_entropy_cleaning=False,
        precision=32,
        mz_shift_aug_p=0.0,
        mz_shift_aug_max=0.0,
    )


def preprocess_spectra_batch(
    spectra_np: np.ndarray,
    spec_preproc: du.SpectrumPreprocessor,
    prec_mz_np: np.ndarray | None = None,
) -> np.ndarray:
    batch_spec_proc = []
    for i, spec in enumerate(spectra_np):
        spec_arr = np.asarray(spec, dtype=np.float32)
        if spec_arr.ndim != 2:
            raise ValueError(f"Expected single spectrum with 2 dims, got shape={spec_arr.shape}")
        if spec_arr.shape[1] == 2:
            high_form = True
        elif spec_arr.shape[0] == 2:
            high_form = False
        else:
            raise ValueError(f"Spectrum is not in shape (n_peaks, 2) or (2, n_peaks); got {spec_arr.shape}")

        prec = None
        if prec_mz_np is not None:
            p = float(prec_mz_np[i])
            if np.isfinite(p):
                prec = p
        batch_spec_proc.append(spec_preproc(spec_arr, prec_mz=prec, high_form=high_form, augment=False))
    return np.asarray(batch_spec_proc, dtype=np.float32)


def predict_batch(
    model: FingerprintHead,
    batch_spec_np: np.ndarray,
    batch_charge_np: np.ndarray,
    device: str,
    apply_sigmoid_to_pred: bool,
    amp_dtype: str = "none",
) -> np.ndarray:
    batch_spec = torch.tensor(np.asarray(batch_spec_np, dtype=np.float32), device=device)
    batch_charge = torch.tensor(np.asarray(batch_charge_np, dtype=np.float32), device=device)

    use_amp = device == "cuda" and amp_dtype != "none"
    amp_torch_dtype = getattr(torch, amp_dtype) if use_amp else None

    with torch.inference_mode():
        with (
            torch.autocast(device_type="cuda", dtype=amp_torch_dtype)
            if use_amp
            else nullcontext()
        ):
            pred = model(batch_spec, batch_charge)
        if apply_sigmoid_to_pred:
            pred = torch.sigmoid(pred)
    return pred.float().detach().cpu().numpy().astype(np.float32)


def load_reference_val_batch(
    finetuning_hdf5: Path,
    fp_kind: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    msdata = du.MSData(finetuning_hdf5, in_mem=False)
    spec_preproc = build_train_like_spec_preprocessor()
    dataset = msdata.to_torch_dataset(
        spec_preproc=spec_preproc,
        label=f"fp_{fp_kind}",
        dformat=DataFormatBuilder("A").get_dformat(),
    )

    split_mask = pd.Series(msdata.get_values(FOLD)).apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )
    if (split_mask == "val").sum() == 0 and (split_mask == "test").sum() > 0:
        split_mask = split_mask.replace({"test": "val"})

    datamodule = du.SplittedDataModule(
        dataset=dataset,
        split_mask=split_mask,
        batch_size=batch_size,
        num_workers=0,
        n_train_samples=None,
        seed=1,
    )
    loader = datamodule.val_dataloader()
    if loader is None:
        raise RuntimeError("Regression check could not build a validation dataloader.")

    batch = next(iter(loader))
    return (
        batch["spec"].detach().cpu().numpy().astype(np.float32),
        batch["charge"].detach().cpu().numpy().astype(np.float32),
    )


def run_validation_regression_check(
    model: FingerprintHead,
    spec_val: np.ndarray,
    prec_val: np.ndarray,
    finetuning_hdf5: Path,
    fp_kind: str,
    device: str,
    apply_sigmoid_to_pred: bool,
    amp_dtype: str,
    batch_size: int,
    reference_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
) -> dict:
    cache_key = (fp_kind, batch_size)
    if cache_key not in reference_cache:
        reference_cache[cache_key] = load_reference_val_batch(
            finetuning_hdf5=finetuning_hdf5,
            fp_kind=fp_kind,
            batch_size=batch_size,
        )

    ref_spec_np, ref_charge_np = reference_cache[cache_key]
    script_spec_np = preprocess_spectra_batch(
        spectra_np=spec_val[:batch_size],
        spec_preproc=build_train_like_spec_preprocessor(),
        prec_mz_np=prec_val[:batch_size],
    )
    script_charge_np = np.ones((batch_size,), dtype=np.float32)

    spec_max_abs = float(np.max(np.abs(ref_spec_np - script_spec_np)))
    charge_max_abs = float(np.max(np.abs(ref_charge_np - script_charge_np)))

    ref_pred_np = predict_batch(
        model=model,
        batch_spec_np=ref_spec_np,
        batch_charge_np=ref_charge_np,
        device=device,
        apply_sigmoid_to_pred=apply_sigmoid_to_pred,
        amp_dtype="none",
    )
    script_pred_np = predict_batch(
        model=model,
        batch_spec_np=script_spec_np,
        batch_charge_np=script_charge_np,
        device=device,
        apply_sigmoid_to_pred=apply_sigmoid_to_pred,
        amp_dtype=amp_dtype,
    )

    pred_max_abs = float(np.max(np.abs(ref_pred_np - script_pred_np)))
    pred_mean_abs = float(np.mean(np.abs(ref_pred_np - script_pred_np)))
    pred_ok = np.allclose(ref_pred_np, script_pred_np, rtol=1e-5, atol=1e-4)
    spec_ok = np.allclose(ref_spec_np, script_spec_np, rtol=0.0, atol=1e-6)
    charge_ok = np.allclose(ref_charge_np, script_charge_np, rtol=0.0, atol=1e-6)

    summary = {
        "batch_size": int(batch_size),
        "spec_max_abs": spec_max_abs,
        "charge_max_abs": charge_max_abs,
        "pred_max_abs": pred_max_abs,
        "pred_mean_abs": pred_mean_abs,
        "passed": bool(spec_ok and charge_ok and pred_ok),
    }

    if not summary["passed"]:
        raise RuntimeError(
            "Validation regression check failed. "
            f"spec_max_abs={spec_max_abs:.6g}, charge_max_abs={charge_max_abs:.6g}, "
            f"pred_max_abs={pred_max_abs:.6g}, pred_mean_abs={pred_mean_abs:.6g}, "
            f"amp_dtype={amp_dtype}. "
            "The script preprocessing/model path no longer matches the training dataloader path."
        )

    return summary


def infer_batches(
    model: FingerprintHead,
    spectra_np: np.ndarray,
    batch_size: int,
    device: str,
    apply_sigmoid_to_pred: bool,
    progress_desc: str,
    prec_mz_np: np.ndarray | None = None,
    amp_dtype: str = "none",
) -> np.ndarray:
    outputs = []
    spec_preproc = build_train_like_spec_preprocessor()

    for start in tqdm(
        range(0, len(spectra_np), batch_size),
        desc=progress_desc,
        leave=True,
        dynamic_ncols=True,
    ):
        end = min(start + batch_size, len(spectra_np))
        batch_spec_np = preprocess_spectra_batch(
            spectra_np=spectra_np[start:end],
            spec_preproc=spec_preproc,
            prec_mz_np=None if prec_mz_np is None else prec_mz_np[start:end],
        )
        batch_charge_np = np.ones((end - start,), dtype=np.float32)
        outputs.append(
            predict_batch(
                model=model,
                batch_spec_np=batch_spec_np,
                batch_charge_np=batch_charge_np,
                device=device,
                apply_sigmoid_to_pred=apply_sigmoid_to_pred,
                amp_dtype=amp_dtype,
            )
        )

    return np.concatenate(outputs, axis=0)


def save_predictions(
    run_dir: Path,
    y_pred_ood: np.ndarray,
    y_pred_val: np.ndarray,
    metadata: dict,
) -> None:
    axis2_dir = run_dir / "axis2_artifacts"
    axis2_dir.mkdir(parents=True, exist_ok=True)

    np.save(axis2_dir / "y_pred.npy", y_pred_ood)
    np.save(axis2_dir / "y_pred_val.npy", y_pred_val)
    (axis2_dir / "inference_only_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")

    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")

    specs = load_specs(args.specs_json)

    if args.run_tags:
        selected = {x.strip() for x in args.run_tags.split(",") if x.strip()}
        specs = [s for s in specs if s["run_tag"] in selected]

    if not specs:
        raise ValueError("No model specs selected.")

    spec_ood, spec_val, prec_ood, prec_val = load_datasets(args.probing_test, args.finetuning_hdf5)

    precision_label = f"amp={args.amp_dtype}" if args.amp_dtype != "none" else "float32"
    print(
        f"Running {len(specs)} checkpoint(s) on device={args.device}, "
        f"batch_size={args.batch_size}, precision={precision_label}"
    )
    reference_batch_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for idx, spec in enumerate(tqdm(specs, desc="Model runs", dynamic_ncols=True), start=1):
        run_tag = spec["run_tag"]

        ckpt_file = spec.get("ckpt_file")
        ckpt_path = args.ckpt_base_dir / ckpt_file if ckpt_file else None

        if ckpt_path is None or not ckpt_path.exists():
            ckpt_path = discover_checkpoint_for_spec(args.ckpt_base_dir, spec)
            print(f"Auto-resolved checkpoint for {run_tag}: {ckpt_path}")

        run_dir = args.output_root / run_tag
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoints_dir / "best.ckpt"
        if ckpt_path.resolve() != best_ckpt_path.resolve():
            shutil.copy2(ckpt_path, best_ckpt_path)

        print(f"\n[{idx}/{len(specs)}] {run_tag}")
        print(f"Checkpoint: {best_ckpt_path}")

        model = load_model_compat(best_ckpt_path, args.device)
        regression_summary = None

        if not args.skip_regression_check:
            check_batch_size = min(args.regression_check_batch_size, len(spec_val))
            print(f"Running regression check on first {check_batch_size} validation samples...")
            regression_summary = run_validation_regression_check(
                model=model,
                spec_val=spec_val,
                prec_val=prec_val,
                finetuning_hdf5=args.finetuning_hdf5,
                fp_kind=spec["fp_kind"],
                device=args.device,
                apply_sigmoid_to_pred=bool(spec["apply_sigmoid_to_pred"]),
                amp_dtype=args.amp_dtype,
                batch_size=check_batch_size,
                reference_cache=reference_batch_cache,
            )
            print(
                "Regression check passed: "
                f"spec_max_abs={regression_summary['spec_max_abs']:.3g}, "
                f"pred_max_abs={regression_summary['pred_max_abs']:.3g}, "
                f"pred_mean_abs={regression_summary['pred_mean_abs']:.3g}"
            )

        t0 = time.perf_counter()
        y_pred_ood = infer_batches(
            model,
            spec_ood,
            batch_size=args.batch_size,
            device=args.device,
            apply_sigmoid_to_pred=bool(spec["apply_sigmoid_to_pred"]),
            progress_desc=f"{run_tag} | OOD inference",
            prec_mz_np=prec_ood,
            amp_dtype=args.amp_dtype,
        )
        t1 = time.perf_counter()
        y_pred_val = infer_batches(
            model,
            spec_val,
            batch_size=args.batch_size,
            device=args.device,
            apply_sigmoid_to_pred=bool(spec["apply_sigmoid_to_pred"]),
            progress_desc=f"{run_tag} | VAL inference",
            prec_mz_np=prec_val,
            amp_dtype=args.amp_dtype,
        )
        t2 = time.perf_counter()

        metadata = {
            "run_tag": run_tag,
            "checkpoint": str(best_ckpt_path),
            "fp_kind": spec["fp_kind"],
            "loss_kind": spec["loss_kind"],
            "apply_sigmoid_to_pred": bool(spec["apply_sigmoid_to_pred"]),
            "device": args.device,
            "precision_mode": precision_label,
            "amp_dtype": args.amp_dtype,
            "batch_size": args.batch_size,
            "y_pred_ood_shape": list(y_pred_ood.shape),
            "y_pred_val_shape": list(y_pred_val.shape),
            "seconds_ood": t1 - t0,
            "seconds_val": t2 - t1,
            "seconds_total": t2 - t0,
        }
        if regression_summary is not None:
            metadata["regression_check"] = regression_summary
        save_predictions(run_dir, y_pred_ood, y_pred_val, metadata)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Saved predictions for {run_tag}: "
            f"OOD {y_pred_ood.shape}, VAL {y_pred_val.shape}, "
            f"total {metadata['seconds_total']:.1f}s"
        )

    print("\nAll selected runs finished.")


if __name__ == "__main__":
    main()
