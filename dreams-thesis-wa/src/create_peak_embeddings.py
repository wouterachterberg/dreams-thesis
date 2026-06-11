#!/usr/bin/env python3
"""
Create or update finetuning_with_ssl_embeddings.hdf5 with full peak-level SSL embeddings.

Adds:
- ssl_peak_embeddings: [N, P, 1024]
Optionally adds:
- ssl_peak_mask: [N, P] uint8 (1=valid peak, 0=padding)

Keeps existing datasets (including ssl_embedding [N, 1024]) unchanged.
Verifies consistency between ssl_embedding and ssl_peak_embeddings[:, 0, :].

Usage:
    python dreams-thesis-wa/src/create_peak_embeddings.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS
from dreams.definitions import PRETRAINED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract full SSL peak embeddings to HDF5.")
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5",
        help="Target HDF5 with spectrum dataset and existing ssl_embedding.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--n-highest-peaks", type=int, default=100, help="Expected DreaMS input peak count.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for ssl_peak_embeddings.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Comma-separated subset of folds to process (e.g. train,val) or 'all'.",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ssl_peak_embeddings dataset.")
    parser.add_argument(
        "--write-peak-mask",
        action="store_true",
        help="Also write ssl_peak_mask [N,P] where padded peaks are 0.",
    )
    return parser.parse_args()


def choose_device(user_device: str | None) -> str:
    if user_device:
        if user_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        if user_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps requested but MPS is not available.")
        return user_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_fold(v: np.ndarray) -> np.ndarray:
    if len(v) == 0:
        return np.asarray(v, dtype=object)
    if isinstance(v[0], (bytes, np.bytes_)):
        return np.asarray([x.decode("utf-8") for x in v], dtype=object)
    return np.asarray(v, dtype=object)


def ensure_model_input_layout(spec: np.ndarray, n_highest_peaks: int) -> np.ndarray:
    arr = np.asarray(spec, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [B,2,P] or [B,P,2], got {arr.shape}")

    if arr.shape[-1] == 2:
        model_in = arr
    elif arr.shape[1] == 2:
        model_in = np.transpose(arr, (0, 2, 1)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported spectrum layout: {arr.shape}")

    p = model_in.shape[1]
    if p == n_highest_peaks:
        return model_in
    if p > n_highest_peaks:
        return model_in[:, :n_highest_peaks, :]

    out = np.zeros((model_in.shape[0], n_highest_peaks, 2), dtype=np.float32)
    out[:, :p, :] = model_in
    return out


def load_ssl_model(device: str, n_highest_peaks: int):
    ckpt = PRETRAINED / "ssl_model.ckpt"

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
        model = PreTrainedModel.from_ckpt(
            ckpt_path=ckpt,
            ckpt_cls=DreaMS,
            n_highest_peaks=n_highest_peaks,
            remove_unused_backbone_parameters=True,
        )
    finally:
        torch.load = original_torch_load

    model.model.eval()
    model.model = model.model.to(device)
    return model


def compute_valid_peak_mask(model_in: np.ndarray) -> np.ndarray:
    # model_in layout: [B, P, 2], channel 1 is intensity.
    ints = model_in[..., 1]
    return (ints > 0).astype(np.uint8)


def main() -> None:
    args = parse_args()

    if not args.hdf5.exists():
        raise FileNotFoundError(f"HDF5 not found: {args.hdf5}")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading SSL model from: {PRETRAINED / 'ssl_model.ckpt'}")

    ssl_model = load_ssl_model(device=device, n_highest_peaks=args.n_highest_peaks)
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    with h5py.File(args.hdf5, "r+") as f:
        if "spectrum" not in f:
            raise KeyError("Expected spectrum dataset in HDF5.")

        n_rows = int(f["spectrum"].shape[0])
        print(f"Rows: {n_rows:,}")

        fold_mask = np.ones((n_rows,), dtype=bool)
        if args.folds.lower() != "all":
            if "fold" not in f:
                raise KeyError("fold dataset missing but --folds subset requested.")
            folds_req = {x.strip() for x in args.folds.split(",") if x.strip()}
            fold_vec = decode_fold(f["fold"][:])
            fold_mask = np.array([str(x) in folds_req for x in fold_vec], dtype=bool)
            print(f"Processing folds={sorted(folds_req)} => {int(fold_mask.sum()):,}/{n_rows:,} rows")

        if "ssl_peak_embeddings" in f:
            if not args.overwrite:
                raise FileExistsError(
                    "ssl_peak_embeddings already exists. Use --overwrite to regenerate."
                )
            del f["ssl_peak_embeddings"]

        if args.write_peak_mask and "ssl_peak_mask" in f:
            if args.overwrite:
                del f["ssl_peak_mask"]
            else:
                raise FileExistsError("ssl_peak_mask already exists. Use --overwrite to regenerate.")

        # Infer peak count by passing one mini-batch through layout normalization.
        sample = f["spectrum"][0:1].astype(np.float32)
        sample_in = ensure_model_input_layout(sample, args.n_highest_peaks)
        p = int(sample_in.shape[1])

        peak_ds = f.create_dataset(
            "ssl_peak_embeddings",
            shape=(n_rows, p, 1024),
            dtype=out_dtype,
            compression="gzip",
        )
        peak_ds.attrs["description"] = "Frozen SSL embeddings per peak token"
        peak_ds.attrs["source_ckpt"] = str(PRETRAINED / "ssl_model.ckpt")
        peak_ds.attrs["n_highest_peaks"] = int(args.n_highest_peaks)
        peak_ds.attrs["storage_dtype"] = args.dtype

        mask_ds = None
        if args.write_peak_mask:
            mask_ds = f.create_dataset(
                "ssl_peak_mask",
                shape=(n_rows, p),
                dtype=np.uint8,
                compression="gzip",
            )
            mask_ds.attrs["description"] = "1 for non-padding peaks, 0 for padded peaks"

        ssl_prec = f["ssl_embedding"] if "ssl_embedding" in f else None
        max_abs_diff = 0.0
        checked = 0

        for start in tqdm(range(0, n_rows, args.batch_size), desc="Extracting peak embeddings"):
            end = min(start + args.batch_size, n_rows)
            idx = np.arange(start, end)
            active = fold_mask[idx]

            if not np.any(active):
                continue

            active_idx = idx[active]
            spectrum_batch = f["spectrum"][active_idx].astype(np.float32)
            model_in = ensure_model_input_layout(spectrum_batch, args.n_highest_peaks)

            x = torch.from_numpy(model_in).to(device)
            with torch.no_grad():
                embs = ssl_model.model(x).detach().cpu().numpy().astype(np.float32)

            peak_ds[active_idx] = embs.astype(out_dtype)

            if mask_ds is not None:
                mask_ds[active_idx] = compute_valid_peak_mask(model_in)

            if ssl_prec is not None:
                prec_ref = ssl_prec[active_idx].astype(np.float32)
                prec_new = embs[:, 0, :]
                batch_diff = float(np.max(np.abs(prec_ref - prec_new)))
                max_abs_diff = max(max_abs_diff, batch_diff)
                checked += len(active_idx)

        print("Done writing ssl_peak_embeddings.")
        if ssl_prec is not None and checked > 0:
            print(f"Precursor consistency check rows: {checked:,}")
            print(f"Max abs diff ssl_embedding vs ssl_peak_embeddings[:,0,:]: {max_abs_diff:.6e}")

    print(f"Updated HDF5: {args.hdf5}")


if __name__ == "__main__":
    main()
