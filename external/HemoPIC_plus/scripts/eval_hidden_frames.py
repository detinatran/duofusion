"""Point prediction of hidden time frames by the physics models, to compare with
the neural operator (scripts/train_operator.py --hide-blocks).

python scripts/eval_hidden_frames.py configs/full_aif.yaml [--blocks 5] [--workers 30]

As eval_heldout.py, the frames of every ROI curve are split into --blocks
contiguous blocks and each block in turn is hidden (weight 0). The model is
refitted on the remaining frames and predicts the hidden ones. Error of a block:
weighted squared error on the hidden frames over the weighted signal energy of
the whole curve, so the blocks of one ROI add up to a curve-level error.
Two starts are reported:
  cold: from the pipeline's prior start (fit_roi_curve), three Pe starts for the
        PDE; uses nothing of the hidden frames;
  warm: from the fit on all frames (as eval_heldout.py).
Models: the compartment (<name>), the plain transport PDE (one tube, its fit
theta_pde in npde_theta.npz) and the neural PDE (<name>__npde, closure of the
held-out fold frozen, its two-tube split kept fixed). The ROI descriptors of
the closure (and of the operator) come from the whole curve in both cases.
Writes <out_root>/<name>__npde/eval/hidden_frames.csv.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_heldout import fit_masked  # noqa: E402
from run_npde_uq import PE_STARTS, prior_start  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde.uqbatch import PDEBatch, load_npde_subject  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import objective, simulate  # noqa: E402


def block_rows(b, model, name, blocks, starts):
    theta_all = LP.refine_map(b)
    B, T = b.c_obs.shape
    edges = np.linspace(0, T, blocks + 1).astype(int)
    rows = []
    for i in range(B):
        energy = float((b.w[i] * b.c_obs[i] ** 2).sum())
        for k in range(blocks):
            hide = torch.zeros(T, dtype=torch.bool)
            hide[edges[k]:edges[k + 1]] = True
            w = b.w[i].clone()
            w[hide] = 0.0
            row = {"subject": name, "roi_id": int(b.roi_ids[i]), "tissue": b.tissue[i], "model": model, "block": k,
                   "n_hidden": int(hide.sum()), "energy": energy}
            for label, th0s in (("cold", starts[i]), ("warm", [theta_all[i]])):
                best, best_obj = None, None
                for th0 in th0s:
                    th0 = torch.minimum(torch.maximum(th0, b.lo[i]), b.hi[i])
                    th = fit_masked(b, i, th0, w)
                    with torch.no_grad():
                        ob = float(objective(b, th[None, :], roi_index=[i], w=w[None, :])[0][0])
                    if best is None or ob < best_obj:
                        best, best_obj = th, ob
                with torch.no_grad():
                    _, c_fit = simulate(b, best[None, :], roi_index=[i])
                r2 = b.w[i] * (c_fit[0] - b.c_obs[i]) ** 2
                row[f"err_hidden_{label}"] = float(r2[hide].sum()) / max(energy, 1e-30)
                row[f"err_visible_{label}"] = float(r2[~hide].sum()) / max(energy, 1e-30)
            rows.append(row)
    return rows


def run_subject(args):
    fit_dir, nd, blocks, models = args
    torch.set_num_threads(1)
    t0 = time.time()
    rows = []
    if models is None or "compartment" in models:
        tank = load_subject(fit_dir)
        th = prior_start(tank)[:, :tank.lo.shape[1]]
        rows += block_rows(tank, "compartment", fit_dir.name, blocks, [[th[i]] for i in range(len(tank.roi_ids))])
        if nd is None:
            print(f"  {fit_dir.name}: {len(tank.roi_ids)} ROIs, {time.time() - t0:.0f}s", flush=True)
            return rows
    pb = load_npde_subject(fit_dir, nd)
    starts = [[] for _ in pb.roi_ids]
    for pe in PE_STARTS:
        th = prior_start(pb)
        th[:, 7] = float(np.log(pe))
        for i in range(len(pb.roi_ids)):
            starts[i].append(th[i])
    if models is None or "neural PDE" in models:
        rows += block_rows(pb, "neural PDE", fit_dir.name, blocks, starts)
    z = np.load(nd / "npde_theta.npz")
    if "theta_pde" in z and (models is None or "PDE" in models):
        pp = PDEBatch(pb.b, z["theta_pde"], None, None, pb.n_cells)
        rows += block_rows(pp, "PDE", fit_dir.name, blocks, starts)
    print(f"  {fit_dir.name}: {len(pb.roi_ids)} ROIs, {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--models", nargs="*", help="subset of: compartment, PDE, 'neural PDE' (default all)")
    p.add_argument("--output", default="hidden_frames.csv")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("out_root", a.out_root),) if v})
    root = cfg.resolve("out_root") / cfg.name
    npde_root = root.parent / f"{cfg.name}__npde"
    tank_only = a.models == ["compartment"]
    pids = a.patients or list_patients(root if tank_only else npde_root)
    jobs = [(root / subject_id(pid), None if tank_only else npde_root / subject_id(pid), a.blocks, a.models) for pid in pids
            if (tank_only or (npde_root / subject_id(pid) / "npde_theta.npz").exists()) and (root / subject_id(pid) / "fit_curves.npz").exists()]
    workers = a.workers or os.cpu_count() or 1
    print(f"[{cfg.name}] hidden-frame prediction on {len(jobs)} subjects, {a.blocks} blocks, {workers} workers")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run_subject, jobs):
            rows.extend(r)
    out = (root if tank_only else npde_root) / "eval"
    out.mkdir(exist_ok=True)
    with open(out / a.output, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{cfg.name}] done in {time.time() - t0:.0f}s")
    for model in ("compartment", "PDE", "neural PDE"):
        sel = [r for r in rows if r["model"] == model]
        if not sel:
            continue
        for label in ("cold", "warm"):
            per_roi = {}
            for r in sel:
                per_roi.setdefault((r["subject"], r["roi_id"]), 0.0)
                per_roi[(r["subject"], r["roi_id"])] += r[f"err_hidden_{label}"]
            print(f"  {model:12s} {label}: hidden-frame error per ROI (sum over blocks), median {np.median(list(per_roi.values())):.4f}")


if __name__ == "__main__":
    main()
