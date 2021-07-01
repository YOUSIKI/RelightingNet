import torch
import numpy as np

__all__ = [
    'load_checkpoint',
]


def load_checkpoint(module, ckpt):
    if isinstance(ckpt, str):
        ckpt = np.load(ckpt)
    for m in module.modules():
        if hasattr(m, 'load_checkpoint'):
            m.load_checkpoint(ckpt)