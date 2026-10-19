"""AIF uncertainty: rerun the pipeline with perturbed arterial input functions and
measure how much the maps, the lesion AUC and the absolute values move.

python scripts/run_aif_ensemble.py configs/full_aif.yaml [--members 10] [--workers 8]

Member k = 1..M bootstraps the selected arterial voxels and jitters the
partial-volume correction (hemopic_plus/aif.py); the last two members also keep
the second earliest shape cluster. Member 0 is the experiment itself
(<out_root>/<name>/training_XX). oSVD is recomputed with every member's AIF, so
the model - oSVD difference is checked member by member. Only patients whose own
AIF passed QC are used; a member whose AIF fails QC (the pipeline then falls back
to the tissue mean) is left out for that patient.
Absolute values across patients: for the brain and lesion medians of every map,
the between-patient spread (log scale) against the spread across AIF members of
one patient (intraclass correlation, rank correlation of the patients with the
AIF itself), and the lesion ROC AUC of one absolute threshold shared by all
patients (voxels pooled over the cohort, a fixed sample per patient) against the
same with each map divided by its brain median.
Writes <out_root>/<name>__aif/m<k>/training_XX/ (fits),
       <out_root>/<name>/training_XX/aif_ens/HemoPIC_{CBF,CBV,MTT}_aif_cv.nii.gz
         (voxel coefficient of variation across members of the map relative to its brain median),
       <out_root>/<name>/eval_uq/aif_ensemble.{csv,json}.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, resolve_dt, save_nifti_like, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.pipeline import run_subject  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402


def member_cfg(cfg, k, n_members):
    return replace(cfg, aif_member=k, aif_cluster_rank=1 if k > n_members - 2 else 0)


def fit_dir(out, name, k, sub):
    return out / name / sub if k == 0 else out / f"{name}__aif" / f"m{k}" / sub


def auc(vol, brain, ot, direction):
    valid, score = _lesion_scores(vol, brain, direction)
    return roc_auc(ot[valid], score) if score is not None else np.nan


def work(args):
    k, pid, cfg, a = args
    t0 = time.time()
    out, name, sub = cfg.resolve("out_root"), cfg.name, subject_id(pid)
    fd = fit_dir(out, name, k, sub)
    if k > 0 and not (a.resume and (fd / "HemoPIC_MTT.nii.gz").exists()):
        try:
            run_subject(member_cfg(cfg, k, a.members), pid, fd.parent)
        except Exception as e:  # noqa: BLE001
            return {"member": k, "patient": pid, "error": repr(e)}
    info = json.loads((fd / "run_info_core.json").read_text(encoding="utf8"))
    row = {"member": k, "patient": pid, "arterial_input": info["arterial_input"],
           "pve_scale": (info.get("aif") or {}).get("pve_scale", np.nan),
           "arterial_auc": (info.get("prior_calibration") or {}).get("arterial_auc", np.nan)}
    if info["arterial_input"] != "auto_aif":
        return row
    base = cfg.resolve("data_root") / sub
    img, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg = load_nifti(base / SEG4_REL)[1].astype(int)
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    brain = seg > 0
    show = brain & np.all(np.isfinite(ctc), axis=3)
    rng = np.random.default_rng(pid)
    les, rest = np.flatnonzero(show & ot), np.flatnonzero(show & ~ot)
    pick = np.r_[rng.choice(les, min(les.size, a.sample // 4), replace=False),
                 rng.choice(rest, min(rest.size, a.sample - a.sample // 4), replace=False)]
    row["_lesion"] = ot.ravel()[pick]

    def levels(src, m, vol):
        ok = brain & np.isfinite(vol) & (vol > 0)
        row[f"{src} median {m}"] = float(np.median(vol[ok])) if ok.any() else np.nan
        row[f"{src} lesion median {m}"] = float(np.median(vol[ok & ot])) if (ok & ot).any() else np.nan
        row[f"_{src} {m}"] = vol.ravel()[pick]
    for m, direction in MAPS:
        vol = nib.load(str(fd / f"HemoPIC_{m}.nii.gz")).get_fdata()
        row[f"model {m}"] = auc(vol, brain, ot, direction)
        levels("model", m, vol)
    dt, _ = resolve_dt(cfg, img, pid)
    art = positive_excursion(np.load(fd / "fit_curves.npz")["cart_norm"][None, :], cfg.baseline_frames)[0]
    curves = positive_excursion(ctc[show].astype(np.float64), cfg.baseline_frames)
    o = dict(zip(("CBF", "CBV", "MTT"), osvd_maps(curves, art, dt, k_h=cfg.k_h, rho=cfg.rho_brain)))
    for m, direction in MAPS:
        vol = np.full(seg.shape, np.nan)
        vol[show] = o[m]
        row[f"oSVD {m}"] = auc(vol, brain, ot, direction)
        levels("oSVD", m, vol)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def voxel_spread(args):
    pid, cfg, members = args
    out, name, sub = cfg.resolve("out_root"), cfg.name, subject_id(pid)
    base = cfg.resolve("data_root") / sub
    seg_img, seg = load_nifti(base / SEG4_REL)
    brain = seg > 0
    res = {"patient": pid, "n_members": len(members)}
    od = out / name / sub / "aif_ens"
    od.mkdir(parents=True, exist_ok=True)
    for m, _ in MAPS:
        rel = []
        for k in members:
            v = nib.load(str(fit_dir(out, name, k, sub) / f"HemoPIC_{m}.nii.gz")).get_fdata()
            ok = brain & np.isfinite(v) & (v > 0)
            rel.append(np.where(ok, v / np.median(v[ok]), np.nan))
        rel = np.stack(rel)
        cv = np.nanstd(rel, axis=0) / np.nanmean(rel, axis=0)
        save_nifti_like(seg_img, np.where(brain, cv, np.nan), od / f"HemoPIC_{m}_aif_cv.nii.gz")
        res[f"voxel cv {m}"] = float(np.nanmedian(cv[brain]))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--resume", action="store_true", help="reuse member fits that already exist")
    p.add_argument("--sample", type=int, default=4000, help="voxels per patient for the pooled AUC (a quarter in the lesion)")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, name = cfg.resolve("out_root"), cfg.name
    pids = sorted(a.patients or list_patients(cfg.resolve("data_root")))
    pids = [p_ for p_ in pids if json.loads((out / name / subject_id(p_) / "run_info_core.json").read_text())["arterial_input"] == "auto_aif"]
    jobs = [(k, pid, cfg, a) for pid in pids for k in range(a.members + 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, jobs):
            rows.append(r)
            if "error" in r:
                print(f"  m{r['member']} training_{r['patient']}: {r['error']}", flush=True)
            elif r["member"] == a.members:
                print(f"  training_{r['patient']} done", flush=True)
    valid = {}
    for r in rows:
        if r.get("arterial_input") == "auto_aif" and "model CBF" in r:
            valid.setdefault(r["patient"], {})[r["member"]] = r
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        spread = {r["patient"]: r for r in ex.map(voxel_spread, [(p_, cfg, sorted(valid[p_])) for p_ in sorted(valid)])}
    od = out / name / "eval_uq"
    od.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r if not k.startswith("_")})
    with open(od / "aif_ensemble.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def cv(x):
        x = np.asarray([v for v in x if np.isfinite(v)])
        return float(np.std(x) / np.mean(x)) if x.size > 1 else np.nan

    pts = sorted(valid)
    members = range(a.members + 1)
    summ = {"members": a.members, "n_patients": len(pts),
            "n_member_fallbacks": int(sum(1 for r in rows if r.get("arterial_input") == "tissue_mean")),
            "n_errors": int(sum(1 for r in rows if "error" in r))}
    for src in ("model", "oSVD"):
        for m, _ in MAPS:
            per_member = [float(np.nanmean([valid[p_][k][f"{src} {m}"] for p_ in pts if k in valid[p_]])) for k in members]
            within = [float(np.std([valid[p_][k][f"{src} {m}"] for k in valid[p_]])) for p_ in pts]
            summ[f"AUC {src} {m}"] = {"member0": per_member[0], "min": min(per_member), "max": max(per_member),
                                      "per_member": per_member, "mean_within_patient_sd": float(np.mean(within))}
    for m, _ in MAPS:
        diffs = []
        for k in members:
            d = np.array([valid[p_][k][f"model {m}"] - valid[p_][k][f"oSVD {m}"] for p_ in pts if k in valid[p_]])
            diffs.append({"member": k, "diff": float(np.mean(d)), "p": float(wilcoxon(d).pvalue)})
        summ[f"model - oSVD {m}"] = diffs
    level_keys = [f"{src} {reg} {m}" for src in ("model", "oSVD") for reg in ("median", "lesion median") for m, _ in MAPS]
    for key in (*level_keys, "arterial_auc", "pve_scale"):
        summ[f"CV across members, {key}"] = float(np.median([cv([valid[p_][k][key] for k in valid[p_]]) for p_ in pts]))
    full = [p_ for p_ in pts if all(k in valid[p_] for k in members)]
    summ["n_patients_all_members"] = len(full)
    for key in level_keys:
        X = np.log(np.array([[valid[p_][k][key] for k in members] for p_ in full]))
        X = X[np.all(np.isfinite(X), axis=1)]
        K = X.shape[1]
        msb, msw = K * X.mean(axis=1).var(ddof=1), X.var(axis=1, ddof=1).mean()
        rho = [float(spearmanr(X[:, 0], X[:, k])[0]) for k in range(1, K)]
        summ[f"across patients, {key}"] = {
            "n": int(X.shape[0]), "between-patient sd of log, AIF itself": float(X[:, 0].std(ddof=1)),
            "within-patient sd of log across AIFs": float(np.sqrt(msw)), "ICC": float((msb - msw) / (msb + (K - 1) * msw)),
            "rank correlation with the AIF itself, median": float(np.median(rho)), "min": float(np.min(rho))}
    for src in ("model", "oSVD"):
        for m, direction in MAPS:
            for scale in ("absolute", "relative"):
                per_member = []
                for k in members:
                    ys, ss = [], []
                    for p_ in full:
                        r = valid[p_][k]
                        v = r[f"_{src} {m}"] / (r[f"{src} median {m}"] if scale == "relative" else 1.0)
                        ok = np.isfinite(v) & (v > 0)
                        ys.append(r["_lesion"][ok])
                        ss.append(1.0 / v[ok] if direction == "low" else v[ok])
                    per_member.append(roc_auc(np.concatenate(ys), np.concatenate(ss)))
                summ[f"pooled AUC, {src} {m}, {scale}"] = {"member0": per_member[0], "min": min(per_member),
                                                         "max": max(per_member), "per_member": per_member}
    for m, _ in MAPS:
        summ[f"median voxel CV of the relative map, {m}"] = float(np.median([spread[p_][f"voxel cv {m}"] for p_ in pts]))
    (od / "aif_ensemble.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(f"{len(pts)} patients, {a.members} AIF members (+ the AIF itself); member fallbacks {summ['n_member_fallbacks']}")
    for src in ("model", "oSVD"):
        for m, _ in MAPS:
            s = summ[f"AUC {src} {m}"]
            print(f"  AUC {src:5s} {m}: AIF itself {s['member0']:.3f}, members {s['min']:.3f}-{s['max']:.3f}, "
                  f"within-patient sd {s['mean_within_patient_sd']:.3f}")
    for m, _ in MAPS:
        ds = summ[f"model - oSVD {m}"]
        print(f"  model - oSVD {m}: " + " ".join(f"{d['diff']:+.3f}" for d in ds) + f"  (p < 0.05 in {sum(d['p'] < 0.05 for d in ds)}/{len(ds)})")
    for k, v in summ.items():
        if k.startswith("CV across") or k.startswith("median voxel CV"):
            print(f"  {k}: {v:.3f}")
    print(f"across {len(full)} patients with every member (log scale)")
    for key in level_keys:
        s = summ[f"across patients, {key}"]
        print(f"  {key:26s} between sd {s['between-patient sd of log, AIF itself']:.3f}, within sd "
              f"{s['within-patient sd of log across AIFs']:.3f}, ICC {s['ICC']:.2f}, rank corr {s['rank correlation with the AIF itself, median']:.2f} "
              f"(min {s['min']:.2f})")
    print("pooled lesion AUC, one threshold for all patients")
    for src in ("model", "oSVD"):
        for m, _ in MAPS:
            sa, sr = summ[f"pooled AUC, {src} {m}, absolute"], summ[f"pooled AUC, {src} {m}, relative"]
            print(f"  {src:5s} {m}: absolute {sa['member0']:.3f} ({sa['min']:.3f}-{sa['max']:.3f}), "
                  f"relative {sr['member0']:.3f} ({sr['min']:.3f}-{sr['max']:.3f})")


if __name__ == "__main__":
    main()
