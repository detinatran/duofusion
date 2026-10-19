"""Load one fitted subject (outputs/<config>/training_XX/) as a RoiBatch."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from .torch_model import DTYPE, RoiBatch, bounds_for, weight_from_curve


def config_from_run_info(info):
    cfg = Config()
    for k, v in info.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_subject(fit_dir: Path, only_success=True):
    fit_dir = Path(fit_dir)
    info = json.loads((fit_dir / "run_info_core.json").read_text(encoding="utf8"))
    cfg = config_from_run_info(info)
    z = np.load(fit_dir / "fit_curves.npz")
    t = z["t"].astype(np.float64)
    dt = float(z["dt"]) if "dt" in z else float(info["dt_seconds"])
    shape = z["cart_norm"].astype(np.float64)
    w_curve = z["w_curve"].astype(np.float64) if "w_curve" in z else shape
    w = weight_from_curve(w_curve)

    rows = {}
    with open(fit_dir / "roi_fit_summary.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            rows[int(r["roi_id"])] = r

    keep, theta, lo, hi, f0s, v0s, c0s, tissue = [], [], [], [], [], [], [], []
    for i, rid in enumerate(z["roi_ids"]):
        r = rows[int(rid)]
        if only_success and not bool(z["success"][i]):
            continue
        f0, v0, c0 = float(r["F_prior"]), float(r["V_prior"]), float(r["Cin_prior"])
        lo_i, hi_i = bounds_for(cfg, t, dt, f0, v0, c0)
        th = [np.log(float(r["R_fit"])), np.log(float(r["tau_fit"])), np.log(float(r["V_fit"])),
              np.log(float(r["F_fit"])), np.log(float(r["Cin_bar_fit"]))]
        if cfg.delay_dispersion:
            th += [float(r["delay_fit"]), np.log(max(float(r["dispersion_fit"]), 1e-12))]
        keep.append(i)
        theta.append(th)
        lo.append(lo_i)
        hi.append(hi_i)
        f0s.append(max(f0, 1e-6))
        v0s.append(max(v0, 1e-6))
        c0s.append(max(c0, 1e-6))
        tissue.append(r["tissue"])
    def T(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=DTYPE)
    c_obs = z["c_obs"][keep].astype(np.float64)
    batch = RoiBatch(
        roi_ids=z["roi_ids"][keep].astype(int), tissue=tissue, c_obs=T(c_obs),
        w=T(np.broadcast_to(w, c_obs.shape).copy()), shape=T(shape), t=T(t), dt=dt,
        log_f0=T(np.log(f0s)), log_v0=T(np.log(v0s)), log_cin0=T(np.log(c0s)),
        lo=T(lo), hi=T(hi), theta_fit=T(theta), cfg=cfg)
    batch.c_fit_numpy = z["c_fit"][keep].astype(np.float64)
    batch.w_curve_raw = w_curve
    batch.info = info
    return batch
