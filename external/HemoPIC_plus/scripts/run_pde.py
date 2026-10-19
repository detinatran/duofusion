"""Refit every ROI with the transport-PDE tissue model and evaluate it.

python scripts/run_pde.py configs/full.yaml --data-root ../HemoPIC/data/ISLES2017_Training [--cells 24]

Starts from the HemoPIC(+) fit of run_cohort.py (needs delay_dispersion: true).
For every subject three variants are written, each as a complete fit folder
that scripts/evaluate.py-style evaluation can read:

  <out_root>/<name>__polish/   the same tank model, Powell solution polished with
                               L-BFGS-B (separates optimiser gains from model gains)
  <out_root>/<name>__pde/      tank replaced by the convection-dispersion PDE
                               (log Pe added; same priors; Pe -> 0 is HemoPIC+)

Maps are the fitted maps rescaled ROI by ROI with the new CBF level and V
(as in run_uq.py); MTT = 60 CBV / CBF with the pipeline's floors and caps.
Writes <out_root>/<name>__pde/eval/summary.json and prints a comparison.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, REGIONS, evaluate_subject  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, derived, simulate  # noqa: E402
from hemopic_plus.uq.torch_model import objective as tank_objective  # noqa: E402

PE_STARTS = (0.02, 0.3, 3.0, 30.0)


def fit_pde(b, theta0, n_cells, maxiter=300, paths=None):
    """L-BFGS-B on all ROIs of a subject at once (the objective is a sum of
    independent terms), restarted from several Pe; the best start is kept per ROI."""
    lo, hi = TP.bounds(b)
    B, P = theta0.shape[0], 8
    best_th, best_obj = None, None
    for pe in PE_STARTS:
        x0 = torch.cat([theta0[:, :7], torch.full((B, 1), float(np.log(pe)), dtype=DTYPE)], 1)
        x0 = torch.minimum(torch.maximum(x0, lo), hi)

        def f(x):
            th = torch.as_tensor(x, dtype=DTYPE).reshape(B, P).requires_grad_(True)
            val = TP.objective(b, th, n_cells, paths)[0].sum()
            val.backward()
            return float(val.detach()), th.grad.detach().numpy().ravel().astype(np.float64)

        res = minimize(f, x0.numpy().ravel(), jac=True, method="L-BFGS-B",
                       bounds=list(zip(lo.numpy().ravel(), hi.numpy().ravel())), options={"maxiter": maxiter})
        th = torch.as_tensor(res.x, dtype=DTYPE).reshape(B, P)
        with torch.no_grad():
            obj = TP.objective(b, th, n_cells, paths)[0]
        if best_th is None:
            best_th, best_obj = th.clone(), obj.clone()
        else:
            better = obj < best_obj
            best_th[better], best_obj[better] = th[better], obj[better]
    return best_th, best_obj


def rescaled_maps(src, roi_ids, ratio_cbf, ratio_v, cfg):
    """The fitted maps of src rescaled ROI by ROI (CBF level, and V unless CBV
    comes from indicator dilution per voxel); MTT = 60 CBV / CBF with the
    pipeline's floors and caps. Returns (image, cbf, cbv, mtt)."""
    lab = nib.load(str(src / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
    img = nib.load(str(src / "HemoPIC_CBF.nii.gz"))
    cbf = img.get_fdata().astype(np.float64)
    cbv = nib.load(str(src / "HemoPIC_CBV.nii.gz")).get_fdata().astype(np.float64)
    for k, rid in enumerate(roi_ids):
        m = lab == int(rid)
        cbf[m] *= ratio_cbf[k]
        if cfg.cbv_source != "id_voxel":
            cbv[m] *= ratio_v[k]
    show = np.isfinite(cbf)
    info = json.loads((src / "run_info_core.json").read_text(encoding="utf8"))
    if cfg.map_bounds == "reference_maps":
        cbf_floor, cbf_cap = info["cbf_floor_from_gt_p1"], info["cbf_cap_from_gt_p99_5"]
        mtt_floor, mtt_cap = info["mtt_floor_from_gt_p1"], info["mtt_cap_from_gt_p99_5"]
    else:
        ok = show & (cbf > 0)
        cbf_floor, cbf_cap = float(np.percentile(cbf[ok], 1.0)), float(np.percentile(cbf[ok], 99.5))
        mtt_floor, mtt_cap = float(cfg.mtt_fixed[0]), float(cfg.mtt_fixed[1])
    den = np.minimum(np.maximum(cbf, max(cbf_floor, cfg.eps_small)), max(cbf_cap, cbf_floor))
    mtt = np.full_like(cbf, np.nan)
    ok = show & np.isfinite(cbv) & (cbv > 0) & (den > 0)
    mtt[ok] = 60.0 * cbv[ok] / den[ok]
    mtt = np.where(show, np.minimum(np.maximum(mtt, mtt_floor), mtt_cap), np.nan)
    return img, cbf, cbv, mtt


def write_variant(src, dst, b, c_new, ratio_cbf, ratio_v, cfg):
    """Copy the fit folder and replace curves and maps."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("roi_labels_seg4_within_kmeans.nii.gz", "roi_fit_summary.csv", "run_info_core.json"):
        shutil.copy2(src / name, dst / name)
    z = dict(np.load(src / "fit_curves.npz"))
    pos = {int(r): i for i, r in enumerate(z["roi_ids"])}
    c_fit = z["c_fit"].copy()
    for k, rid in enumerate(b.roi_ids):
        c_fit[pos[int(rid)]] = c_new[k]
    z["c_fit"] = c_fit
    np.savez_compressed(dst / "fit_curves.npz", **z)
    img, cbf, cbv, mtt = rescaled_maps(src, b.roi_ids, ratio_cbf, ratio_v, cfg)
    for tag, vol in (("CBF", cbf), ("CBV", cbv), ("MTT", mtt)):
        nib.save(nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header), str(dst / f"HemoPIC_{tag}.nii.gz"))


def run_subject(args):
    fit_dir, root, name, n_cells = args
    torch.set_num_threads(1)
    t0 = time.time()
    b = load_subject(fit_dir)
    cfg = b.cfg
    with torch.no_grad():
        ref = {k: v.numpy() for k, v in derived(b, b.theta_fit).items()}
    th_pol = LP.refine_map(b)
    with torch.no_grad():
        _, c_pol = simulate(b, th_pol)
        d_pol = {k: v.numpy() for k, v in derived(b, th_pol).items()}
        obj_tank = tank_objective(b, th_pol)[0].numpy()
    th_pde, obj_pde = fit_pde(b, th_pol, n_cells)
    with torch.no_grad():
        _, c_pde = TP.simulate(b, th_pde, n_cells)
        d_pde = {k: v.numpy() for k, v in derived(b, th_pde[:, :7]).items()}
        def nm(c):
            return ((b.w * (c - b.c_obs) ** 2).mean(1) / (b.w * b.c_obs ** 2).mean(1)).numpy()
        data_pol, data_pde = nm(c_pol), nm(c_pde)
    sub = fit_dir.name
    write_variant(fit_dir, root.parent / f"{name}__polish" / sub, b, c_pol.numpy(),
                  d_pol["CBF"] / ref["CBF"], d_pol["V"] / ref["V"], cfg)
    write_variant(fit_dir, root.parent / f"{name}__pde" / sub, b, c_pde.numpy(),
                  d_pde["CBF"] / ref["CBF"], d_pde["V"] / ref["V"], cfg)
    rows = []
    for k, rid in enumerate(b.roi_ids):
        rows.append({"subject": sub, "roi_id": int(rid), "tissue": b.tissue[k],
                     "obj_tank": float(obj_tank[k]), "obj_pde": float(obj_pde[k]),
                     "nmse_tank": float(data_pol[k]), "nmse_pde": float(data_pde[k]),
                     "Pe": float(np.exp(th_pde[k, 7])), "delay_tank": float(th_pol[k, 5]), "delay_pde": float(th_pde[k, 5]),
                     "disp_tank": float(np.exp(th_pol[k, 6])), "disp_pde": float(np.exp(th_pde[k, 6])),
                     **{f"{q}_tank": float(d_pol[q][k]) for q in ("CBF", "V", "MTT", "tau")},
                     **{f"{q}_pde": float(d_pde[q][k]) for q in ("CBF", "V", "MTT", "tau")}})
    np.savez_compressed(root.parent / f"{name}__pde" / sub / "pde_theta.npz", roi_ids=b.roi_ids,
                        theta=th_pde.numpy(), theta_tank=th_pol.numpy(), n_cells=n_cells)
    return rows, time.time() - t0


def evaluate(data_root, fit_root, pids, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_eval, [(data_root, fit_root, p) for p in pids]))
    rows = [r for rr, _ in res for r in rr]
    out = {}
    for reg in REGIONS:
        v = [r["value"] for r in rows if r["kind"] == "tracer" and r["region"] == reg]
        out[f"nRMSE_{reg}"] = float(np.nanmean(v))
    for m, _ in MAPS:
        v = [r["value"] for r in rows if r["kind"] == "roc" and r["region"] == m and r["method"] == "model"]
        out[f"AUC_{m}"] = float(np.nanmean(v))
    per = {(r["patient"], r["kind"], r["region"]): r["value"] for r in rows if r["method"] == "model"}
    return out, per


def _eval(args):
    data_root, fit_root, pid = args
    try:
        return evaluate_subject(data_root, fit_root, pid)
    except Exception as e:
        print("  eval skip", pid, repr(e))
        return [], {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--cells", type=int, default=24)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    if not cfg.delay_dispersion:
        raise SystemExit("run_pde.py needs a config with delay_dispersion: true")
    root = cfg.resolve("out_root") / cfg.name
    pids = a.patients or list_patients(root)
    jobs = [(root / subject_id(pid), root, cfg.name, a.cells) for pid in pids
            if (root / subject_id(pid) / "fit_curves.npz").exists()]
    workers = a.workers or os.cpu_count() or 1
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, sec in ex.map(run_subject, jobs):
            rows.extend(r)
    print(f"[{cfg.name}] PDE fits of {len(rows)} ROIs in {len(jobs)} subjects, {time.time() - t0:.0f}s ({a.cells} cells)")
    pde_root = root.parent / f"{cfg.name}__pde"
    (pde_root / "eval").mkdir(parents=True, exist_ok=True)
    with open(pde_root / "eval" / "roi_compare.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    data_root = cfg.resolve("data_root")
    pids_ok = [int(j[0].name.split("_")[1]) for j in jobs]
    summ = {}
    per = {}
    for tag, fr in (("pipeline", root), ("tank_polished", root.parent / f"{cfg.name}__polish"), ("pde", pde_root)):
        summ[tag], per[tag] = evaluate(data_root, fr, pids_ok, workers)
    pe = np.array([r["Pe"] for r in rows])
    gain = np.array([r["nmse_tank"] - r["nmse_pde"] for r in rows])
    summ["roi"] = {"n": len(rows), "Pe_median": float(np.median(pe)), "Pe_quartiles": np.percentile(pe, [25, 75]).tolist(),
                   "frac_Pe_gt_1": float(np.mean(pe > 1.0)),
                   "nmse_tank_median": float(np.median([r["nmse_tank"] for r in rows])),
                   "nmse_pde_median": float(np.median([r["nmse_pde"] for r in rows])),
                   "frac_pde_better": float(np.mean(gain > 0)),
                   **{f"{q}_rel_change_median": float(np.median([abs(r[f"{q}_pde"] / r[f"{q}_tank"] - 1) for r in rows]))
                      for q in ("CBF", "V", "MTT", "tau")}}
    paired = {}
    for key in [("tracer", reg) for reg in REGIONS] + [("roc", m) for m, _ in MAPS]:
        d = [per["pde"].get((pid, *key), np.nan) - per["tank_polished"].get((pid, *key), np.nan) for pid in pids_ok]
        d = np.array(d, float)
        d = d[np.isfinite(d)]
        if d.size:
            from scipy.stats import wilcoxon
            pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            paired[f"{key[0]}_{key[1]}"] = {"mean_diff": float(d.mean()), "p_wilcoxon": pv}
    summ["paired_pde_minus_polished"] = paired
    (pde_root / "eval" / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")

    print(f"  {'':16s} " + " ".join(f"{k:>14s}" for k in summ["pde"]))
    for tag in ("pipeline", "tank_polished", "pde"):
        print(f"  {tag:16s} " + " ".join(f"{v:14.4f}" for v in summ[tag].values()))
    r = summ["roi"]
    print(f"  Pe median {r['Pe_median']:.2f} [IQR {r['Pe_quartiles'][0]:.2f}, {r['Pe_quartiles'][1]:.2f}], Pe > 1 in {r['frac_Pe_gt_1']:.0%} of ROIs;"
          f" weighted nMSE median tank {r['nmse_tank_median']:.4f} -> PDE {r['nmse_pde_median']:.4f}, PDE better in {r['frac_pde_better']:.0%}")
    print("  median |relative change| PDE vs tank: " + ", ".join(f"{q} {r[f'{q}_rel_change_median']:.1%}" for q in ("CBF", "V", "MTT", "tau")))
    print("  paired PDE - tank_polished: " + ", ".join(f"{k} {v['mean_diff']:+.4f} (p={v['p_wilcoxon']:.1g})" for k, v in paired.items()))


if __name__ == "__main__":
    main()
