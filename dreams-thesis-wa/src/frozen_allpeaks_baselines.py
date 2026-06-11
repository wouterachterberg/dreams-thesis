#!/usr/bin/env python3
"""
Train frozen-backbone all-peaks DeepSets baselines.

Input:
- finetuning_with_ssl_embeddings.hdf5 with ssl_peak_embeddings [N,P,1024]
- finetuning.hdf5 with fold labels
- fingerprint_cache.npz with targets

Model:
- DeepSets head identical in form to fine-tuned head: phi -> sum pool -> rho
- Uses explicit peak mask to ignore zero-padded peaks during pooling

Outputs:
- results/frozen_allpeaks_baselines/<run_tag>/checkpoints/*
- history.csv
- run_config.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train all-peaks frozen DeepSets baseline.")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--embedding-hdf5",
        type=Path,
        default=root / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning_with_ssl_embeddings.hdf5",
    )
    parser.add_argument(
        "--finetuning-hdf5",
        type=Path,
        default=root / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5",
    )
    parser.add_argument(
        "--fingerprint-cache",
        type=Path,
        default=root / "dreams-thesis-wa/data/processed/MassSpecGym_splits/fingerprint_cache.npz",
    )
    parser.add_argument("--fp-kind", type=str, required=True, choices=["morgan_2048", "maccs_166", "map4_2048"])
    parser.add_argument("--loss-kind", type=str, required=True, choices=["bce_logits", "cos"])
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=103)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="ssl_peak_embeddings",
        help="3D embedding dataset key in embedding-hdf5",
    )
    parser.add_argument(
        "--use-hdf5-peak-mask",
        action="store_true",
        help="Use ssl_peak_mask from embedding HDF5 if present, else derive from spectrum intensity.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(user_device: Optional[str]) -> torch.device:
    if user_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if user_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps requested but MPS is unavailable")

    if user_device:
        return torch.device(user_device)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_utf8(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.asarray(values, dtype=object)
    if isinstance(values[0], (bytes, np.bytes_)):
        return np.asarray([x.decode("utf-8") for x in values], dtype=object)
    return np.asarray(values, dtype=object)


def load_targets(fp_cache: Path, fp_kind: str):
    z = np.load(fp_cache, mmap_mode="r")
    mapping = {
        "morgan_2048": "morgan_fps",
        "maccs_166": "maccs_fps",
        "map4_2048": "map4_fps",
    }
    k = mapping[fp_kind]
    if k not in z:
        raise KeyError(f"Missing {k} in {fp_cache}")
    return z[k]


class CosSimLoss(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(inputs, targets).mean()


class PeakSetDeepSetsHead(nn.Module):
    def __init__(self, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.Dropout(dropout),
        )
        self.rho = nn.Linear(1024, out_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, P, 1024]
        h = self.phi(x)
        if mask is not None:
            # mask: [B, P] with 1=valid peak, 0=padding.
            h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1)
        return self.rho(pooled)


@dataclass
class TrainConfig:
    fp_kind: str
    loss_kind: str
    run_tag: str
    pos_weight: Optional[float]
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    max_epochs: int
    early_stop_patience: int
    seed: int
    device: str
    embedding_key: str
    use_hdf5_peak_mask: bool


class PeakEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_hdf5: Path,
        embedding_key: str,
        y_all: np.ndarray,
        indices: np.ndarray,
        use_hdf5_peak_mask: bool,
    ):
        self.embedding_hdf5 = embedding_hdf5
        self.embedding_key = embedding_key
        self.y_all = y_all
        self.indices = np.asarray(indices, dtype=np.int64)
        self.use_hdf5_peak_mask = use_hdf5_peak_mask
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.embedding_hdf5, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])

        x = self._h5[self.embedding_key][idx]
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != 1024:
            raise ValueError(f"Expected [P,1024] embedding row, got {x.shape}")

        if self.use_hdf5_peak_mask and "ssl_peak_mask" in self._h5:
            mask = np.asarray(self._h5["ssl_peak_mask"][idx], dtype=np.float32)
        else:
            if "spectrum" not in self._h5:
                raise KeyError("spectrum dataset is required for deriving padding mask")
            spec = np.asarray(self._h5["spectrum"][idx], dtype=np.float32)
            if spec.ndim != 2:
                raise ValueError(f"Unexpected spectrum shape at row {idx}: {spec.shape}")
            if spec.shape[0] == 2:
                ints = spec[1]
            elif spec.shape[1] == 2:
                ints = spec[:, 1]
            else:
                raise ValueError(f"Unsupported spectrum layout at row {idx}: {spec.shape}")
            mask = (ints > 0).astype(np.float32)

        y = np.asarray(self.y_all[idx], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y)

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def make_loss(loss_kind: str, n_bits: int, pos_weight: Optional[float], device: torch.device):
    if loss_kind == "cos":
        return CosSimLoss()
    if loss_kind == "bce_logits":
        if pos_weight is None:
            return nn.BCEWithLogitsLoss()
        pw = torch.full((n_bits,), float(pos_weight), dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    raise ValueError(f"Unsupported loss kind: {loss_kind}")


def eval_epoch(model, loader, criterion, loss_kind, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)
            pred = model(xb, mb)
            pred_for_loss = pred
            if loss_kind in {"bce", "cross_entropy"}:
                pred_for_loss = torch.sigmoid(pred)
            loss = criterion(pred_for_loss, yb)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else np.nan


def train_one(args: argparse.Namespace):
    set_seed(args.seed)
    device = choose_device(args.device)

    with h5py.File(args.finetuning_hdf5, "r") as f:
        fold = decode_utf8(f["fold"][:])

    y_all = load_targets(args.fingerprint_cache, args.fp_kind)

    if len(y_all) != len(fold):
        raise ValueError(f"Target length mismatch: y={len(y_all)} fold={len(fold)}")

    train_idx = np.where(fold == "train")[0]
    val_idx = np.where(fold == "val")[0]
    if len(val_idx) == 0:
        val_idx = np.where(fold == "test")[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Empty train/val split after fold filtering")

    train_ds = PeakEmbeddingDataset(
        embedding_hdf5=args.embedding_hdf5,
        embedding_key=args.embedding_key,
        y_all=y_all,
        indices=train_idx,
        use_hdf5_peak_mask=args.use_hdf5_peak_mask,
    )
    val_ds = PeakEmbeddingDataset(
        embedding_hdf5=args.embedding_hdf5,
        embedding_key=args.embedding_key,
        y_all=y_all,
        indices=val_idx,
        use_hdf5_peak_mask=args.use_hdf5_peak_mask,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    n_bits = int(y_all.shape[1])
    model = PeakSetDeepSetsHead(out_dim=n_bits, dropout=args.dropout).to(device)
    criterion = make_loss(args.loss_kind, n_bits=n_bits, pos_weight=args.pos_weight, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = args.project_root / "dreams-thesis-wa/results/frozen_allpeaks_baselines" / args.run_tag
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    history_rows = []

    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []

        for xb, mb, yb in train_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, mb)
            pred_for_loss = pred
            if args.loss_kind in {"bce", "cross_entropy"}:
                pred_for_loss = torch.sigmoid(pred)
            loss = criterion(pred_for_loss, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = eval_epoch(model, val_loader, criterion, args.loss_kind, device)

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (patience={args.early_stop_patience})")
            break

    if best_state is None:
        raise RuntimeError("No model state captured during training")

    ckpt = {
        "model_state_dict": best_state,
        "epoch": int(best_epoch),
        "val_loss": float(best_val),
        "fp_kind": args.fp_kind,
        "loss_kind": args.loss_kind,
        "run_tag": args.run_tag,
        "input_mode": "all_peaks",
        "embedding_key": args.embedding_key,
        "architecture": "phi->sum(masked)->rho",
    }

    ckpt_best = ckpt_dir / f"{args.run_tag}_best.ckpt"
    ckpt_alias = ckpt_dir / "best.ckpt"
    torch.save(ckpt, ckpt_best)
    torch.save(ckpt, ckpt_alias)

    history = pd.DataFrame(history_rows)
    history.to_csv(run_dir / "history.csv", index=False)

    cfg = TrainConfig(
        fp_kind=args.fp_kind,
        loss_kind=args.loss_kind,
        run_tag=args.run_tag,
        pos_weight=args.pos_weight,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device=str(device),
        embedding_key=args.embedding_key,
        use_hdf5_peak_mask=bool(args.use_hdf5_peak_mask),
    )

    out_cfg = {
        **asdict(cfg),
        "embedding_hdf5": str(args.embedding_hdf5),
        "finetuning_hdf5": str(args.finetuning_hdf5),
        "fingerprint_cache": str(args.fingerprint_cache),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_checkpoint": str(ckpt_best),
        "runtime_sec": float(time.time() - t0),
    }

    (run_dir / "run_config.json").write_text(json.dumps(out_cfg, indent=2))

    print("Done.")
    print(f"run_dir: {run_dir}")
    print(f"best_ckpt: {ckpt_best}")
    print(f"best_epoch: {best_epoch}, best_val_loss: {best_val:.6f}")


def main():
    args = parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
