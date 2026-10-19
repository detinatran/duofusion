"""Is the infarct outside Tmax > 6 s real, or an artefact of registration, borders or Tmax?

python scripts/qa_outside_tmax.py <data_root> <out_dir> [--workers 8] [--figures 8]

Per patient, for the infarct voxels in tissue (OT):
  in_tmax6 / in_tmax4      share inside the admission Tmax > 6 s / > 4 s region
  near_<d>mm               share within d mm (in plane) of the Tmax > 6 s region
  in_contra_tmax6          share inside the mirror image of the Tmax > 6 s region
                           (high = infarct on the wrong side, i.e. a left-right error)
  out_tmax_invalid / out_tmax_0_4 / out_tmax_4_6
                           Tmax of the infarct voxels outside Tmax > 6 s: <= 0 (no
                           measurable bolus, typical of a severe core), 0-4 s, 4-6 s
  out_rcbf_lt30 / out_rcbf_lt50   share of those voxels with CBF below 30% / 50% of the
                           contralateral healthy median (hypoperfused by CBF despite Tmax)
Writes qa_per_patient.csv, qa_summary.json and figures of the patients with the lowest
in_tmax6 (NCCT/CTP mean, Tmax, CBF; red: infarct, yellow: Tmax > 6 s).
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

from duofusion.data import list_subjects, load_subject  # noqa: E402
from duofusion.reference import classic_references, masks  # noqa: E402
from duofusion.symmetry import Mirror  # noqa: E402

NEAR_MM = (3, 6, 10)


def share(a, b):
    return float((a & b).sum() / b.sum()) if b.any() else float("nan")


def work(args):
    root, num = args
    s = load_subject(root, num)
    cbf, tmax = s.maps["CBF"], s.maps["Tmax"]
    valid, _, hypo = masks(s, cbf, tmax)
    y = s.ot & valid
    if not y.any():
        return None
    sp = np.sqrt((np.asarray(s.affine)[:3, :3] ** 2).sum(axis=0))
    # in-plane distance to the Tmax > 6 s region, slice by slice
    dist = np.full(y.shape, np.inf)
    for k in range(y.shape[2]):
        if hypo[..., k].any():
            dist[..., k] = distance_transform_edt(~hypo[..., k], sampling=sp[:2])
    t = np.where(np.isfinite(tmax), tmax, np.nan)
    head = s.seg > 0
    ncct = s.maps.get("NCCT")
    mir = Mirror(s.affine, np.clip(ncct, 0, 80) if ncct is not None else head.astype(np.float32), head)
    contra_hypo = np.nan_to_num(mir(hypo.astype(np.float32), nearest=True, fill=0.0)) > 0.5
    refs, meta = classic_references(s, cbf, tmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cbf / refs["contra_hemi"]
    out_ = y & ~hypo
    row = {"patient": num, "ot_ml": float(y.sum()) * s.voxel_ml, "hypo_ml": float(hypo.sum()) * s.voxel_ml,
           "slices": int(y.shape[2]), "affected_side": meta["affected_side"],
           "in_tmax6": share(hypo, y), "in_tmax4": share(valid & (t > 4), y),
           "in_contra_tmax6": share(contra_hypo & ~hypo, y)}
    for d in NEAR_MM:
        row[f"near_{d}mm"] = share(dist <= d, y)
    with np.errstate(invalid="ignore"):
        row["out_tmax_invalid"] = share(~(t > 0), out_)
        row["out_tmax_0_4"] = share((t > 0) & (t <= 4), out_)
        row["out_tmax_4_6"] = share((t > 4) & (t <= 6), out_)
        row["out_rcbf_lt30"] = share(r < 0.3, out_)
        row["out_rcbf_lt50"] = share(r < 0.5, out_)
    row["registration_metric"] = s.info.get("registration_metric", s.info.get("mirror_metric"))
    return row


def figure(root, num, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = load_subject(root, num)
    cbf, tmax = s.maps["CBF"], s.maps["Tmax"]
    valid, _, hypo = masks(s, cbf, tmax)
    y = s.ot & valid
    ks = np.argsort(-y.sum(axis=(0, 1)))[:3]
    bg = s.maps.get("NCCT", cbf)
    fig, ax = plt.subplots(len(ks), 3, figsize=(9, 3 * len(ks)))
    ax = np.atleast_2d(ax)
    for i, k in enumerate(ks):
        for j, (img, title, vr) in enumerate(((bg, "CT", (0, 80)), (tmax, "Tmax (s)", (0, 12)),
                                             (cbf, "CBF", (0, np.nanpercentile(cbf[valid], 98))))):
            a = ax[i, j]
            a.imshow(np.rot90(img[..., k]), cmap="gray" if j == 0 else "viridis", vmin=vr[0], vmax=vr[1])
            a.contour(np.rot90(y[..., k]), levels=[0.5], colors="red", linewidths=0.8)
            a.contour(np.rot90(hypo[..., k]), levels=[0.5], colors="yellow", linewidths=0.6)
            a.set_title(f"{title}, slice {k}", fontsize=8)
            a.axis("off")
    fig.suptitle(f"patient {num}: red infarct, yellow Tmax > 6 s", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--figures", type=int, default=8)
    a = p.parse_args()
    nums = list_subjects(a.data_root)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(work, [(a.data_root, n) for n in nums]) if r]
    keys = list(rows[0])
    with open(a.out_dir / "qa_per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {"n": len(rows)}
    for k in keys:
        if k in ("patient", "slices", "affected_side"):
            continue
        v = np.array([r[k] for r in rows if r[k] is not None], float)
        v = v[np.isfinite(v)]
        if v.size:
            summary[k] = {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                          "q3": float(np.percentile(v, 75)), "mean": float(v.mean())}
    # voxel-pooled shares (large infarcts weigh more)
    w = np.array([r["ot_ml"] for r in rows])
    for k in ("in_tmax6", "in_tmax4", "near_6mm", "in_contra_tmax6"):
        v = np.array([r[k] for r in rows])
        summary[f"{k}_volume_weighted"] = float(np.nansum(v * w) / w[np.isfinite(v)].sum())
    summary["wrong_side_patients"] = int(sum(r["in_contra_tmax6"] > 0.5 and r["in_tmax6"] < 0.2 for r in rows))
    (a.out_dir / "qa_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    worst = sorted(rows, key=lambda r: r["in_tmax6"])[: a.figures]
    for r in worst:
        figure(a.data_root, r["patient"], a.out_dir / f"qa_patient_{r['patient']}.png")
    print(json.dumps({k: (round(v["median"], 3) if isinstance(v, dict) else v) for k, v in summary.items()}, indent=0))


if __name__ == "__main__":
    main()
