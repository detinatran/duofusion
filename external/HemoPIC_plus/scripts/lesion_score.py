"""Multiparametric lesion score: combine the perfusion maps, the bolus timing, the
left-right asymmetry and the uncertainty of each voxel.

python scripts/lesion_score.py --workers 12

Every map on its own is scored with the lesion ROC AUC. Here a classifier
(logistic regression, and a small MLP) combines several features per voxel and
is evaluated with 5-fold cross-validation by patient, so the score of a patient
comes from a model trained without it. The same features and models are built
from oSVD (same AIF) and from HemoPIC+, on the same patients, for a fair
comparison. Features per voxel:
  maps         log of CBF, CBV, MTT relative to the brain median;
  timing       time to peak of the voxel curve relative to the brain median;
  asymmetry    the same quantities minus their value in the mirrored voxel;
  uncertainty  log of the relative 95% interval width of CBF and MTT (HemoPIC+).
HemoPIC+ maps come from full_aif_idvox_osvd_nc (hand-made, never tuned on lesion
labels); its uncertainty from the posterior of the same ROI fits (full_aif/uq).
Writes <out_root>/lesion_score/{summary.json, per_patient.csv}.
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

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, resolve_dt, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402

MAPS = ("CBF", "CBV", "MTT")


def rel_log(vol, brain):
    ok = brain & np.isfinite(vol) & (vol > 0)
    out = np.full(vol.shape, np.nan)
    if ok.sum() > 100:
        out[ok] = np.log(vol[ok] / np.median(vol[ok]))
    return out


def features(args):
    pid, cfg, out = args
    sub = subject_id(pid)
    base = cfg.resolve("data_root") / sub
    img, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg = load_nifti(base / SEG4_REL)[1].astype(int)
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    dt, _ = resolve_dt(cfg, img, pid)
    brain = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
    curves = positive_excursion(ctc[brain].astype(np.float64), cfg.baseline_frames)
    ttp = np.full(seg.shape, np.nan)
    ttp[brain] = np.argmax(gaussian_filter1d(curves, 1.0, axis=1), axis=1) * dt
    ttp_rel = ttp - np.nanmedian(ttp[brain])
    mirror = lambda x: np.flip(x, axis=0)  # noqa: E731
    sets = {}
    srcs = {"oSVD": (out / "osvd" / sub, "oSVD_{}.nii.gz"), "HemoPIC+": (out / "full_aif_idvox_osvd_nc" / sub, "HemoPIC_{}.nii.gz")}
    for name, (d, pat) in srcs.items():
        if not all((d / pat.format(m)).exists() for m in MAPS):
            continue
        maps = [rel_log(nib.load(str(d / pat.format(m))).get_fdata(), brain) for m in MAPS]
        cols = {f"{m}": v for m, v in zip(MAPS, maps)}
        cols["TTP"] = ttp_rel
        for m, v in zip(MAPS, maps):
            cols[f"{m}_asym"] = v - mirror(v)
        cols["TTP_asym"] = ttp_rel - mirror(ttp_rel)
        if name == "HemoPIC+":
            u = out / "full_aif" / sub / "uq"
            for m in ("CBF", "MTT"):
                fit = nib.load(str(out / "full_aif" / sub / f"HemoPIC_{m}.nii.gz")).get_fdata()
                lo = nib.load(str(u / f"HemoPIC_{m}_lo.nii.gz")).get_fdata()
                hi = nib.load(str(u / f"HemoPIC_{m}_hi.nii.gz")).get_fdata()
                w = np.full(seg.shape, np.nan)
                ok = brain & np.isfinite(fit) & (fit > 0) & np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
                w[ok] = np.log((hi[ok] - lo[ok]) / fit[ok])
                cols[f"{m}_width"] = w
        keys = list(cols)
        X = np.stack([cols[k][brain] for k in keys], axis=1).astype(np.float32)
        sets[name] = (keys, X)
    return pid, ot[brain].astype(np.uint8), sets


def fit_model(X, y, w, kind, seed=0, epochs=300):
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    w = torch.as_tensor(w, dtype=torch.float32)
    if kind == "logreg":
        net = torch.nn.Linear(X.shape[1], 1)
    else:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 16),
                                  torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(X)[:, 0], y, reduction="none") * w).sum() / w.sum()
        loss.backward()
        opt.step()
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/full_aif.yaml"))
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--per-class", type=int, default=3000, help="training voxels per class and patient")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = cfg.resolve("out_root")
    od = out / "lesion_score"
    od.mkdir(exist_ok=True)
    pids = sorted(list_patients(cfg.resolve("data_root")))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        data = {pid: (y, sets) for pid, y, sets in ex.map(features, [(pid, cfg, out) for pid in pids])}
    common = [p for p in pids if all(k in data[p][1] for k in ("oSVD", "HemoPIC+"))]
    groups = {"maps": lambda k: k in MAPS,
              "maps+timing": lambda k: k in MAPS or k == "TTP",
              "maps+timing+asym": lambda k: not k.endswith("_width"),
              "maps+timing+asym+uncertainty": lambda k: True}
    variants = [("oSVD", g) for g in list(groups)[:3]] + [("HemoPIC+", g) for g in groups]
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(common))
    fold = {int(x): i % a.folds for i, x in enumerate(perm)}
    results = {}
    for src, g in variants:
        keys = data[common[0]][1][src][0]
        sel = [i for i, k in enumerate(keys) if groups[g](k)]
        for kind in ("logreg", "mlp"):
            name = f"{src} | {g} | {kind}"
            res = {}
            for k in range(a.folds):
                tr = [p for p in common if fold[p] != k]
                te = [p for p in common if fold[p] == k]
                Xs, ys, ws = [], [], []
                for pid in tr:
                    y, sets = data[pid]
                    X = sets[src][1][:, sel]
                    ok = np.all(np.isfinite(X), axis=1)
                    pos, neg = np.where(ok & (y == 1))[0], np.where(ok & (y == 0))[0]
                    pos = rng.choice(pos, min(a.per_class, pos.size), replace=False) if pos.size else pos
                    neg = rng.choice(neg, min(a.per_class, neg.size), replace=False)
                    for idx, lab in ((pos, 1.0), (neg, 0.0)):
                        if idx.size:
                            Xs.append(X[idx])
                            ys.append(np.full(idx.size, lab))
                            ws.append(np.full(idx.size, 1.0 / idx.size))
                Xtr = np.concatenate(Xs)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                net = fit_model((Xtr - mu) / sd, np.concatenate(ys), np.concatenate(ws), kind, seed=k)
                with torch.no_grad():
                    for pid in te:
                        y, sets = data[pid]
                        X = sets[src][1][:, sel]
                        ok = np.all(np.isfinite(X), axis=1)
                        s = net(torch.as_tensor((X[ok] - mu) / sd, dtype=torch.float32))[:, 0].numpy()
                        res[pid] = roc_auc(y[ok], s)
            results[name] = res
            print(f"  {name:48s} AUC {np.nanmean(list(res.values())):.4f}  (n={len(res)})", flush=True)
    single = {}
    for src in ("oSVD", "HemoPIC+"):
        keys = data[common[0]][1][src][0]
        for m, sign in (("CBF", -1), ("CBV", -1), ("MTT", 1), ("TTP", 1)):
            i = keys.index(m)
            v = {}
            for pid in common:
                y, sets = data[pid]
                x = sets[src][1][:, i]
                ok = np.isfinite(x)
                v[pid] = roc_auc(y[ok], sign * x[ok])
            single[f"{src} {m} alone"] = v
    allres = {**single, **results}
    summ = {k: {"auc": float(np.nanmean(list(v.values()))), "n": len(v)} for k, v in allres.items()}

    def paired(x, y):
        d = np.array([allres[x][p] - allres[y][p] for p in common])
        d = d[np.isfinite(d)]
        return {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}

    tests = {}
    for kind in ("logreg", "mlp"):
        for g in list(groups)[:3]:
            tests[f"HemoPIC+ - oSVD | {g} | {kind}"] = paired(f"HemoPIC+ | {g} | {kind}", f"oSVD | {g} | {kind}")
        tests[f"HemoPIC+ +uncertainty - without | {kind}"] = paired(f"HemoPIC+ | maps+timing+asym+uncertainty | {kind}",
                                                                 f"HemoPIC+ | maps+timing+asym | {kind}")
        tests[f"HemoPIC+ best combo - HemoPIC+ CBF alone | {kind}"] = paired(f"HemoPIC+ | maps+timing+asym+uncertainty | {kind}",
                                                                        "HemoPIC+ CBF alone")
    (od / "summary.json").write_text(json.dumps({"n_patients": len(common), "auc": summ, "paired": tests}, indent=2), encoding="utf8")
    with open(od / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["variant", *common])
        for k, v in allres.items():
            w.writerow([k, *[v.get(p) for p in common]])
    print(f"lesion ROC AUC per patient ({len(common)} patients with oSVD maps, {a.folds}-fold CV by patient)")
    for k, s in summ.items():
        print(f"  {k:52s} {s['auc']:.4f}")
    for k, s in tests.items():
        print(f"  {k:52s} {s['diff']:+.4f}  p={s['p']:.2g}")


if __name__ == "__main__":
    main()
