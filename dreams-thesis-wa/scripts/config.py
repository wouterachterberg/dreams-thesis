from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


REPO_ROOT = env_path("DREAMS_REPO_ROOT", "/Users/wouterachterberg/coding/DreaMS")
THESIS_ROOT = REPO_ROOT / "dreams-thesis-wa"
MODEL_CHECKPOINTS_DIR = env_path(
    "DREAMS_MODEL_CHECKPOINTS_DIR",
    "/Volumes/NVMe_Wouter/THESIS/model_checkpoints",
)
AXIS3_RAW_DATA_DIR = env_path(
    "DREAMS_AXIS3_RAW_DATA_DIR",
    THESIS_ROOT / "data/raw/axis_3_data",
)
INSPECT_CKPT_PATH = env_path(
    "DREAMS_INSPECT_CKPT_PATH",
    "/Volumes/NVMe_Wouter/THESIS/snellius_output/"
    "MorganFingerprints/massspecgym_morgan2048_finetune_20260223_130317/last.ckpt",
)
