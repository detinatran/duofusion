"""Information criteria of the tissue models, per ROI.

python scripts/eval_model_selection.py configs/full_aif.yaml [--workers 24]

For every ROI the compartment fit (<name>), the plain PDE (theta_pde) and the
neural PDE (theta, closure of the held-out fold) are scored with the same data
term (weighted squared error over the weighted signal energy). With Gaussian
residuals and n_eff effectively independent frames (the n_eff of the tempered
posterior, uq/laplace.py), -2 log L = n_eff log(data) + const, so
    AIC = n_eff log(data) + 2 k,   BIC = n_eff log(data) + k log(n_eff),
with k = 7 (compartment) and 8 (PDE, neural PDE; the closure is shared by all
ROIs and its weights are not counted per ROI, noted in the output).
Writes <out_root>/<name>__npde/eval/model_selection.{csv,json}.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde.uqbatch import PDEBatch, load_npde_subject  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import simulate  # noqa: E402

K = {"compartment": 7, "PDE": 8, "neural PDE": 8}


def data_term(b, theta):
    with torch.no_grad():
        _, c = simulate(b, torch.as_tensor(theta, dtype=torch.float64))
    return ((b.w * (c - b.c_obs) ** 2).sum(1) / ((b.w * b.c_obs ** 2).sum(1) + 1e-12)).numpy()


def work(args):
    fit_dir, nd = args
    torch.set_num_threads(1)
    tank = load_subject(fit_dir)
    pb = load_npde_subject(fit_dir, nd)
    z = np.load(nd / "npde_theta.npz")
    d = {"compartment": data_term(tank, LP.refine_map(tank)),
         "PDE": data_term(PDEBatch(tank, z["theta_pde"], None, None, pb.n_cells), z["theta_pde"]),
         "neural PDE": data_term(pb, z["theta"])}
    n = float(LP.n_eff(tank))
    rows = []
    for i, rid in enumerate(tank.roi_ids):
        r = {"subject": fit_dir.name, "roi_id": int(rid), "tissue": tank.tissue[i], "n_eff": n}
        for m, k in K.items():
            dm = max(float(d[m][i]), 1e-12)
            r[f"data {m}"] = dm
            r[f"AIC {m}"] = n * np.log(dm) + 2 * k
            r[f"BIC {m}"] = n * np.log(dm) + k * np.log(n)
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("out_root", a.out_root),) if v})
    root = cfg.resolve("out_root") / cfg.name
    npde_root = root.parent / f"{cfg.name}__npde"
    jobs = [(root / subject_id(pid), npde_root / subject_id(pid)) for pid in list_patients(npde_root)
            if (npde_root / subject_id(pid) / "npde_theta.npz").exists() and (root / subject_id(pid) / "fit_curves.npz").exists()]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, jobs):
            rows += r
    rows = [r for r in rows if r["data compartment"] < 0.5]
    summ = {"n_roi": len(rows), "n_subjects": len(jobs), "k": K,
            "note": "closure weights shared by all ROIs, not counted per ROI"}
    for crit in ("AIC", "BIC"):
        for m in ("PDE", "neural PDE"):
            dlt = np.array([r[f"{crit} {m}"] - r[f"{crit} compartment"] for r in rows])
            summ[f"{crit} {m} - compartment"] = {"median": float(np.median(dlt)), "fraction < 0": float(np.mean(dlt < 0)),
                                                  "fraction < -2": float(np.mean(dlt < -2)), "sum": float(dlt.sum())}
        dlt = np.array([r[f"{crit} neural PDE"] - r[f"{crit} PDE"] for r in rows])
        summ[f"{crit} neural PDE - PDE"] = {"median": float(np.median(dlt)), "fraction < 0": float(np.mean(dlt < 0)),
                                            "fraction < -2": float(np.mean(dlt < -2)), "sum": float(dlt.sum())}
    od = npde_root / "eval"
    od.mkdir(exist_ok=True)
    with open(od / "model_selection.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (od / "model_selection.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
