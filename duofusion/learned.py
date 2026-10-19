"""Learned extrapolation of truncated curves (the network of colab/learned_extrapolation.ipynb).

The network sees a baseline-subtracted curve observed for its first L frames (scaled by its
observed maximum, clamped at 1), the observation mask and the time, and predicts the curve
over TMAX = 70 frames (1 s). Weights: one model per patient fold of the ISLES'24 evaluation
(extrap_fold<k>.pt); external data use the mean of the five models.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TMAX = 70


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n(self.c1(F.gelu(x)))))


class Extrap(nn.Module):
    def __init__(self, c=96):
        super().__init__()
        self.inp = nn.Conv1d(3, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)])
        self.glob = nn.Sequential(nn.Linear(c * TMAX, 256), nn.GELU(), nn.Linear(256, TMAX))
        self.out = nn.Conv1d(c, 1, 1)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        y = self.out(h)[:, 0] + self.glob(h.flatten(1))
        return x[:, 1] * x[:, 0] + (1 - x[:, 1]) * y


_CACHE = {}


def load_models(model_dir, folds=None, device="cpu"):
    """Models for the given folds (all five when None), cached per process."""
    key = (str(model_dir), tuple(folds) if folds is not None else None, device)
    if key not in _CACHE:
        files = sorted(Path(model_dir).glob("extrap_fold*.pt"))
        if folds is not None:
            files = [f for f in files if int(f.stem.replace("extrap_fold", "")) in folds]
        models = []
        for f in files:
            m = Extrap()
            m.load_state_dict(torch.load(f, map_location=device))
            models.append(m.to(device).eval())
        _CACHE[key] = models
    return _CACHE[key]


@torch.no_grad()
def extend(curves, models, device="cpu", batch=32768):
    """(N, T) baseline-subtracted curves -> (N, TMAX): measured frames kept, the rest predicted
    (mean of the models)."""
    curves = np.asarray(curves, np.float32)
    N, T = curves.shape
    if T >= TMAX:
        return curves[:, :TMAX].copy()
    out = np.empty((N, TMAX), np.float32)
    t = torch.arange(TMAX, device=device)[None, :]
    obs = (t < T).float()
    for s in range(0, N, batch):
        c = torch.zeros((min(batch, N - s), TMAX), device=device)
        c[:, :T] = torch.as_tensor(curves[s:s + batch], device=device)
        scale = c.amax(1).clamp(min=1.0)
        x = torch.stack([c / scale[:, None], obs.expand(c.shape[0], -1),
                         t.expand(c.shape[0], -1).float() / TMAX], 1)
        pred = torch.stack([m(x) for m in models]).mean(0) * scale[:, None]
        out[s:s + batch] = pred.cpu().numpy()
    out[:, :T] = curves
    return out
