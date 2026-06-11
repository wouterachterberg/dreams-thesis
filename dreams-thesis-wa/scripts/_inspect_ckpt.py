#!/usr/bin/env python3
"""Quick script to inspect the checkpoint structure."""
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import INSPECT_CKPT_PATH

ckpt_path = INSPECT_CKPT_PATH
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('Top-level keys:', list(ckpt.keys()))

if 'hyper_parameters' in ckpt:
    hp = ckpt['hyper_parameters']
    print('\nHyper parameters keys:', list(hp.keys()))
    for k, v in hp.items():
        if k != 'backbone':
            print(f'  {k}: {v}')
        else:
            print(f'  backbone: <Path>')

if 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    print(f'\nState dict ({len(sd)} keys):')
    for i, (k, v) in enumerate(sd.items()):
        if i < 25:
            print(f'  {k}: {v.shape}')
    head_keys = [k for k in sd.keys() if 'head' in k]
    print(f'\nHead keys ({len(head_keys)}):')
    for k in head_keys:
        print(f'  {k}: {sd[k].shape}')

if 'epoch' in ckpt:
    print(f'\nEpoch: {ckpt["epoch"]}')
if 'global_step' in ckpt:
    print(f'Global step: {ckpt["global_step"]}')
