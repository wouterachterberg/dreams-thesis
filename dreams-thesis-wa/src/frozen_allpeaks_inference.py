#!/usr/bin/env python3
"""
Inference/export script for frozen all-peaks heads.

Produces for each run_tag:
- results/model_runs/<run_tag>/axis2_artifacts/y_pred.npy
- results/model_runs/<run_tag>/axis2_artifacts/y_pred_val.npy
- results/model_runs/<run_tag>/axis2_artifacts/y_true.npy
- results/model_runs/<run_tag>/axis2_artifacts/y_true_val.npy
- results/model_runs/<run_tag>/axis2_artifacts/run_config.json

If probing_test parquet already contains peak embeddings + masks, those are used.
Otherwise OOD peak embeddings are computed on the fly via frozen SSL model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

try:
    from map4 import MAP4
    MAP4_AVAILABLE = True
except Exception:
    MAP4_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS
from dreams.definitions import PRETRAINED


RUN_TAGS_DEFAULT = [
    "morgan_2048_bce_frozen_allpeaks",
    "maccs_166_bce_frozen_allpeaks",
    "map4_2048_bce_frozen_allpeaks",
    "morgan_2048_cos_frozen_allpeaks",
    "maccs_166_cos_frozen_allpeaks",
    "map4_2048_cos_frozen_allpeaks",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference for frozen all-peaks heads.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    p.add_argument(
        "--embedding-hdf5",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5",
    )
    p.add_argument(
        "--finetuning-hdf5",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5",
    )
    p.add_argument(
        "--probing-test",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet",
    )
    p.add_argument(
        "--frozen-allpeaks-root",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/results/frozen_allpeaks_baselines",
    )
    p.add_argument(
        "--model-runs-root",
        type=Path,
        default=PROJECT_ROOT / "dreams-thesis-wa/results/model_runs",
    )
    p.add_argument(
        "--run-tags",
        type=str,
        default=",".join(RUN_TAGS_DEFAULT),
        help="Comma-separated run tags.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None)
    return p.parse_args()


def choose_device(user_device: str | None) -> str:
    if user_device:
        if user_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        if user_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps requested but MPS unavailable")
        return user_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PeakSetDeepSetsHead(nn.Module):
    def __init__(self, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.Dropout(dropout),
        )
        self.rho = nn.Linear(1024, out_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x can be [B,1024] or [B,P,1024].
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.phi(x)
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)
        return self.rho(h.sum(dim=1))


def parse_spectrum_strings(mzs_str, intens_str, n_peaks=128):
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


def ensure_model_input_layout(arr: np.ndarray, n_highest_peaks: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [B,2,P] or [B,P,2], got {x.shape}")

    if x.shape[-1] == 2:
        out = x
    elif x.shape[1] == 2:
        out = np.transpose(x, (0, 2, 1)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported spectrum layout: {x.shape}")

    p = out.shape[1]
    if p == n_highest_peaks:
        return out
    if p > n_highest_peaks:
        return out[:, :n_highest_peaks, :]

    padded = np.zeros((out.shape[0], n_highest_peaks, 2), dtype=np.float32)
    padded[:, :p, :] = out
    return padded


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


def infer_peak_embeddings_for_ood(df_ood: pd.DataFrame, ssl_model, device: str, n_highest_peaks: int, batch_size: int):
    spectra = np.stack(
        [parse_spectrum_strings(m, i) for m, i in zip(df_ood["mzs"], df_ood["intensities"])],
        axis=0,
    ).astype(np.float32)
    model_in = ensure_model_input_layout(spectra, n_highest_peaks=n_highest_peaks)
    masks = (model_in[..., 1] > 0).astype(np.float32)

    embs = []
    with torch.no_grad():
        for s in tqdm(range(0, len(model_in), batch_size), desc="OOD SSL forward"):
            e = min(s + batch_size, len(model_in))
            xb = torch.from_numpy(model_in[s:e]).to(device)
            eb = ssl_model.model(xb).detach().cpu().numpy().astype(np.float32)
            embs.append(eb)

    return np.concatenate(embs, axis=0), masks


def load_head_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint object type: {type(ckpt)}")

    if "rho.weight" not in state:
        raise KeyError("Checkpoint missing rho.weight")
    out_dim = int(state["rho.weight"].shape[0])

    model = PeakSetDeepSetsHead(out_dim=out_dim, dropout=0.0)
    model.load_state_dict(state, strict=False)
    model = model.eval().to(device)
    return model, ckpt


def infer_with_head(model, emb: np.ndarray, mask: np.ndarray | None, apply_sigmoid: bool, batch_size: int, device: str):
    out = []
    with torch.no_grad():
        for s in tqdm(range(0, len(emb), batch_size), desc="Head inference"):
            e = min(s + batch_size, len(emb))
            xb = torch.from_numpy(emb[s:e].astype(np.float32)).to(device)
            mb = None
            if mask is not None:
                mb = torch.from_numpy(mask[s:e].astype(np.float32)).to(device)
            pred = model(xb, mb) if mb is not None else model(xb)
            if apply_sigmoid:
                pred = torch.sigmoid(pred)
            out.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


_map4_cache = {}
def _map4_calc_for_dim(dim):
    if dim not in _map4_cache:
        _map4_cache[dim] = MAP4(dimensions=dim, radius=2)
    return _map4_cache[dim]


def fp_from_smiles(smiles, fp_kind):
    mol = Chem.MolFromSmiles(smiles)
    if fp_kind == "morgan_2048":
        n_bits = 2048
    elif fp_kind == "maccs_166":
        n_bits = 166
    elif fp_kind == "map4_2048":
        n_bits = 2048
    else:
        raise ValueError(f"Unsupported fp kind: {fp_kind}")

    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    if fp_kind == "morgan_2048":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp, dtype=np.float32)
    if fp_kind == "maccs_166":
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.float32)[1:]
    if fp_kind == "map4_2048":
        if not MAP4_AVAILABLE:
            raise RuntimeError("MAP4 package not available")
        calc = _map4_calc_for_dim(2048)
        x = np.asarray(calc.calculate(mol), dtype=np.float32)
        return (x != 0).astype(np.float32)

    raise ValueError(f"Unsupported fp kind: {fp_kind}")


def infer_fp_kind(run_tag: str) -> str:
    if "morgan" in run_tag:
        return "morgan_2048"
    if "maccs" in run_tag:
        return "maccs_166"
    if "map4" in run_tag:
        return "map4_2048"
    raise ValueError(f"Could not infer fp_kind from run_tag={run_tag}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    run_tags = [x.strip() for x in args.run_tags.split(",") if x.strip()]
    if not run_tags:
        raise ValueError("No run tags selected")

    args.model_runs_root.mkdir(parents=True, exist_ok=True)

    # Load val split embeddings/masks once.
    with h5py.File(args.embedding_hdf5, "r") as f:
        if "fold" in f:
            fold = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["fold"][:]])
        else:
            with h5py.File(args.finetuning_hdf5, "r") as ff:
                fold = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ff["fold"][:]])

        if "smiles" in f:
            smiles_all = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["smiles"][:]])
        else:
            with h5py.File(args.finetuning_hdf5, "r") as ff:
                smiles_all = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ff["smiles"][:]])

        if "ssl_peak_embeddings" not in f:
            raise KeyError("ssl_peak_embeddings missing; run create_peak_embeddings.py first")
        emb_all = f["ssl_peak_embeddings"][:].astype(np.float32)

        if "ssl_peak_mask" in f:
            mask_all = f["ssl_peak_mask"][:].astype(np.float32)
        else:
            with h5py.File(args.finetuning_hdf5, "r") as ff:
                spec = ff["spectrum"][:].astype(np.float32)
            # [N,2,P]
            mask_all = (spec[:, 1, :] > 0).astype(np.float32)

    val_mask = fold == "val"
    if val_mask.sum() == 0:
        val_mask = fold == "test"

    emb_val = emb_all[val_mask]
    m_val = mask_all[val_mask]
    smiles_val = smiles_all[val_mask].tolist()

    print(f"Device: {device}")
    print(f"VAL embeddings: {emb_val.shape}, masks: {m_val.shape}")

    # OOD load: prefer precomputed peak embeddings in parquet.
    df_ood = pd.read_parquet(args.probing_test)
    smiles_ood = df_ood["smiles"].astype(str).tolist()

    emb_ood = None
    m_ood = None
    for k in ["ssl_peak_embeddings", "peak_embeddings"]:
        if k in df_ood.columns:
            emb_ood = np.asarray(list(df_ood[k].values), dtype=np.float32)
            break
    for k in ["ssl_peak_mask", "peak_mask", "embedding_mask"]:
        if k in df_ood.columns:
            m_ood = np.asarray(list(df_ood[k].values), dtype=np.float32)
            break

    if emb_ood is None:
        print("OOD peak embeddings not found in parquet; computing via SSL model...")
        ssl_model = load_ssl_model(device=device, n_highest_peaks=args.n_highest_peaks)
        emb_ood, m_ood = infer_peak_embeddings_for_ood(
            df_ood,
            ssl_model=ssl_model,
            device=device,
            n_highest_peaks=args.n_highest_peaks,
            batch_size=args.batch_size,
        )
        print(f"Computed OOD embeddings: {emb_ood.shape}, masks: {m_ood.shape}")
    else:
        if m_ood is None:
            # derive from zero rows
            m_ood = (np.abs(emb_ood).sum(axis=2) > 0).astype(np.float32)
        print(f"Loaded OOD peak embeddings from parquet: {emb_ood.shape}, masks: {m_ood.shape}")

    # Cache true arrays per fp_kind.
    y_true_cache = {}

    for run_tag in run_tags:
        t0 = time.time()
        fp_kind = infer_fp_kind(run_tag)
        apply_sigmoid = "_bce_" in f"_{run_tag}_" or run_tag.endswith("_bce_frozen_allpeaks")

        ckpt_path = args.frozen_allpeaks_root / run_tag / "checkpoints" / f"{run_tag}_best.ckpt"
        if not ckpt_path.exists():
            # fallback
            ckpt_path = args.frozen_allpeaks_root / run_tag / "checkpoints" / "best.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {run_tag}")

        print("\n" + "=" * 80)
        print(f"Run: {run_tag}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"FP kind: {fp_kind} | apply_sigmoid={apply_sigmoid}")

        model, ckpt_meta = load_head_checkpoint(ckpt_path, device=device)

        y_pred_ood = infer_with_head(model, emb_ood, m_ood, apply_sigmoid, args.batch_size, device)
        y_pred_val = infer_with_head(model, emb_val, m_val, apply_sigmoid, args.batch_size, device)

        if fp_kind not in y_true_cache:
            y_true_ood = np.stack([fp_from_smiles(s, fp_kind) for s in tqdm(smiles_ood, desc=f"GT OOD {fp_kind}")]).astype(np.float32)
            y_true_val = np.stack([fp_from_smiles(s, fp_kind) for s in tqdm(smiles_val, desc=f"GT VAL {fp_kind}")]).astype(np.float32)
            y_true_cache[fp_kind] = (y_true_ood, y_true_val)
        else:
            y_true_ood, y_true_val = y_true_cache[fp_kind]

        run_dir = args.model_runs_root / run_tag
        axis2_dir = run_dir / "axis2_artifacts"
        ckpt_dir = run_dir / "checkpoints"
        axis2_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Reproducible checkpoint copy under model_runs.
        dst_ckpt = ckpt_dir / ckpt_path.name
        if ckpt_path.resolve() != dst_ckpt.resolve():
            import shutil
            shutil.copy2(ckpt_path, dst_ckpt)

        np.save(axis2_dir / "y_pred.npy", y_pred_ood)
        np.save(axis2_dir / "y_true.npy", y_true_ood)
        np.save(axis2_dir / "y_pred_val.npy", y_pred_val)
        np.save(axis2_dir / "y_true_val.npy", y_true_val)
        np.save(axis2_dir / "smiles.npy", np.array(smiles_ood, dtype=object))

        meta = {
            "run_tag": run_tag,
            "checkpoint": str(dst_ckpt),
            "fp_kind": fp_kind,
            "loss_kind": "bce" if apply_sigmoid else "cos",
            "apply_sigmoid_to_pred": bool(apply_sigmoid),
            "model_type": "frozen_head_allpeaks",
            "input_mode": "all_peaks",
            "embedding_source_hdf5": str(args.embedding_hdf5),
            "probing_test_path": str(args.probing_test),
            "y_pred_ood_shape": list(y_pred_ood.shape),
            "y_pred_val_shape": list(y_pred_val.shape),
            "seconds_total": float(time.time() - t0),
        }

        # Include metadata from training checkpoint if present.
        if isinstance(ckpt_meta, dict):
            for key in ["epoch", "val_loss", "architecture", "embedding_key"]:
                if key in ckpt_meta:
                    meta[key] = ckpt_meta[key]

        (axis2_dir / "run_config.json").write_text(json.dumps(meta, indent=2))

        print(f"Saved to {axis2_dir}")
        print(f"OOD pred {y_pred_ood.shape} | VAL pred {y_pred_val.shape}")

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
