#!/usr/bin/env python3
"""
Create finetuning_with_ssl_embeddings.hdf5 from finetuning.hdf5.

The output HDF5 keeps all datasets from the input file and adds one extra dataset:
- ssl_embedding: [N, 1024] precursor embedding from the frozen SSL foundation model.

By default, this script reads:
    dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5
and writes:
    dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

# Allow importing from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS
from dreams.definitions import PRETRAINED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create finetuning HDF5 with SSL embeddings from frozen foundation model."
    )
    parser.add_argument(
        "--input-hdf5",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5",
        help="Path to finetuning HDF5 file.",
    )
    parser.add_argument(
        "--output-hdf5",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5",
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--n-highest-peaks",
        type=int,
        default=100,
        help="Number of peaks expected by the SSL model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda", "mps"],
        help="Device for model inference. Default: auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output HDF5 if it already exists.",
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


def ensure_model_input_layout(spectrum_batch: np.ndarray, n_highest_peaks: int) -> np.ndarray:
    """Return spectra as [B, n_highest_peaks, 2] float32 for DreaMS SSL model."""
    arr = np.asarray(spectrum_batch, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected spectrum batch with shape [B,2,N] or [B,N,2], got {arr.shape}")

    if arr.shape[-1] == 2:
        model_in = arr
    elif arr.shape[1] == 2:
        model_in = np.transpose(arr, (0, 2, 1)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported spectrum layout: {arr.shape}")

    peak_n = model_in.shape[1]
    if peak_n == n_highest_peaks:
        return model_in

    if peak_n > n_highest_peaks:
        return model_in[:, :n_highest_peaks, :]

    padded = np.zeros((model_in.shape[0], n_highest_peaks, 2), dtype=np.float32)
    padded[:, :peak_n, :] = model_in
    return padded


def load_ssl_model(device: str, n_highest_peaks: int):
    ssl_ckpt = PRETRAINED / "ssl_model.ckpt"

    # torch>=2.6 defaults torch.load(weights_only=True), which can fail for older checkpoints.
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
            ckpt_path=ssl_ckpt,
            ckpt_cls=DreaMS,
            n_highest_peaks=n_highest_peaks,
            remove_unused_backbone_parameters=True,
        )
    finally:
        torch.load = original_torch_load

    model.model.eval()
    model.model = model.model.to(device)
    return model


def main() -> None:
    args = parse_args()

    if not args.input_hdf5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {args.input_hdf5}")

    if args.output_hdf5.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_hdf5}\n"
            "Use --overwrite to replace it."
        )

    args.output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    if args.output_hdf5.exists() and args.overwrite:
        os.remove(args.output_hdf5)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading SSL model from: {PRETRAINED / 'ssl_model.ckpt'}")
    ssl_model = load_ssl_model(device=device, n_highest_peaks=args.n_highest_peaks)

    with h5py.File(args.input_hdf5, "r") as f_in, h5py.File(args.output_hdf5, "w") as f_out:
        keys = sorted(list(f_in.keys()))
        if "spectrum" not in keys:
            raise KeyError(f"Input HDF5 missing required 'spectrum' dataset. Keys: {keys}")

        n_rows = int(f_in["spectrum"].shape[0])
        print(f"Input rows: {n_rows:,}")
        print(f"Input keys: {keys}")

        # Copy all original datasets exactly, then append SSL embeddings.
        for key in keys:
            f_in.copy(key, f_out, name=key)

        ssl_ds = None
        for start in tqdm(range(0, n_rows, args.batch_size), desc="Embedding + writing"):
            end = min(start + args.batch_size, n_rows)

            spectrum_batch = f_in["spectrum"][start:end].astype(np.float32)
            model_in = ensure_model_input_layout(spectrum_batch, args.n_highest_peaks)

            x = torch.from_numpy(model_in).to(device)
            with torch.no_grad():
                embs = ssl_model.model(x)
                ssl_emb = embs[:, 0, :].detach().cpu().numpy().astype(np.float32)

            if ssl_ds is None:
                ssl_ds = f_out.create_dataset(
                    "ssl_embedding",
                    shape=(n_rows, ssl_emb.shape[1]),
                    dtype=np.float32,
                    compression="gzip",
                )

            ssl_ds[start:end] = ssl_emb

    print("Done.")
    print(f"Saved: {args.output_hdf5}")


if __name__ == "__main__":
    main()
