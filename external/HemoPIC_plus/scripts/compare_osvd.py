"""PDE and neural PDE against oSVD with the same automatic AIF.

python scripts/compare_osvd.py --data-root ../HemoPIC/data/ISLES2017_Training

Needs osvd_check.py (oSVD maps), run_pde.py and train_npde.py outputs. On the
subjects with an accepted AIF (those with oSVD maps), compares the lesion ROC
AUC of CBF / CBV / MTT and the between-patient agreement of brain-mean MTT with
oSVD for:
  * the experiments' own maps: full_aif, full_aif__pde, full_aif__npde (and full*);
  * the best map construction (full_aif_idvox_osvd_nc: CBV by indicator dilution
    per voxel, oSVD voxel pattern for CBF) with the ROI CBF levels of the PDE and
    neural PDE fits, written to full_aif_idvox_osvd_nc__pde / __npde.
Writes <out_root>/osvd/compare_npde.{csv,json} and prints a table with paired
Wilcoxon tests against oSVD.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr, wilcoxon  # noqa: E402

from run_pde import rescaled_maps  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, subject_id  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, derived  # noqa: E402

BEST = "full_aif_idvox_osvd_nc"


def best_with_levels(out, sub, src_variant):
    """BEST maps with the ROI CBF levels of a PDE / neural PDE fit of full_aif."""
    fd = out / BEST / sub
    dst = out / f"{BEST}__{src_variant}" / sub
    th_file = out / f"full_aif__{src_variant}" / sub / ("npde_theta.npz" if src_variant == "npde" else "pde_theta.npz")
    if not (fd.exists() and th_file.exists()):
        return None
    b = load_subject(fd)
    z = np.load(th_file)
    pos = {int(r): i for i, r in enumerate(z["roi_ids"])}
    theta = torch.as_tensor(np.stack([z["theta"][pos[int(r)]][:7] for r in b.roi_ids]), dtype=DTYPE)
    with torch.no_grad():
        ref = derived(b, b.theta_fit)["CBF"].numpy()
        new = derived(b, theta)["CBF"].numpy()
    img, cbf, cbv, mtt = rescaled_maps(fd, b.roi_ids, new / ref, np.ones_like(ref), b.cfg)
    dst.mkdir(parents=True, exist_ok=True)
    for tag, vol in (("CBF", cbf), ("CBV", cbv), ("MTT", mtt)):
        nib.save(nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header), str(dst / f"HemoPIC_{tag}.nii.gz"))
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/full_aif.yaml"))
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, data_root = cfg.resolve("out_root"), cfg.resolve("data_root")
    pids = sorted(int(d.name.split("_")[1]) for d in (out / "osvd").glob("training_*") if (d / "oSVD_CBF.nii.gz").exists())
    variants = ["osvd", "full", "full__pde", "full__npde", "full_aif", "full_aif__pde", "full_aif__npde",
                BEST, f"{BEST}__pde", f"{BEST}__npde"]
    auc = {v: {m: {} for m, _ in MAPS} for v in variants}
    mtt_brain = {v: {} for v in variants}
    for pid in pids:
        sub = subject_id(pid)
        for sv in ("pde", "npde"):
            best_with_levels(out, sub, sv)
        seg = nib.load(str(data_root / sub / SEG4_REL)).get_fdata() > 0
        ot = nib.load(str(find_nii(data_root / sub, "OT"))).get_fdata() > 0
        for v in variants:
            d = out / v / sub
            for m, direction in MAPS:
                f = d / (f"oSVD_{m}.nii.gz" if v == "osvd" else f"HemoPIC_{m}.nii.gz")
                if not f.exists():
                    continue
                vol = nib.load(str(f)).get_fdata()
                valid, score = _lesion_scores(vol, seg, direction)
                auc[v][m][pid] = roc_auc(ot[valid], score) if score is not None else np.nan
                if m == "MTT":
                    x = vol[seg & np.isfinite(vol) & (vol > 0)]
                    mtt_brain[v][pid] = float(np.mean(x)) if x.size else np.nan

    rows, summ = [], {"n_patients": len(pids), "variants": {}}
    for v in variants:
        if not auc[v]["CBF"]:
            continue
        s = {"n": len(auc[v]["CBF"])}
        for m, _ in MAPS:
            common = [pid for pid in pids if pid in auc[v][m] and pid in auc["osvd"][m]]
            x = np.array([auc[v][m][pid] for pid in common])
            y = np.array([auc["osvd"][m][pid] for pid in common])
            ok = np.isfinite(x) & np.isfinite(y)
            s[f"AUC_{m}"] = float(np.mean(x[ok]))
            if v != "osvd":
                d = x[ok] - y[ok]
                s[f"d_{m}_vs_osvd"] = float(d.mean())
                s[f"p_{m}_vs_osvd"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        common = [pid for pid in pids if pid in mtt_brain[v] and pid in mtt_brain["osvd"]]
        if v != "osvd" and len(common) > 3:
            xm = np.array([mtt_brain[v][pid] for pid in common])
            ym = np.array([mtt_brain["osvd"][pid] for pid in common])
            ok = np.isfinite(xm) & np.isfinite(ym)
            s["mtt_brain_r_vs_osvd"] = float(pearsonr(xm[ok], ym[ok])[0])
        summ["variants"][v] = s
        for pid in pids:
            rows.append({"patient": pid, "variant": v, **{f"AUC_{m}": auc[v][m].get(pid, np.nan) for m, _ in MAPS},
                         "mtt_brain": mtt_brain[v].get(pid, np.nan)})
    od = out / "osvd"
    with open(od / "compare_npde.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (od / "compare_npde.json").write_text(json.dumps(summ, indent=2), encoding="utf8")

    print(f"lesion ROC AUC on the {len(pids)} patients with an accepted AIF (difference vs oSVD, Wilcoxon p)")
    for v, s in summ["variants"].items():
        cells = []
        for m, _ in MAPS:
            c = f"{m} {s[f'AUC_{m}']:.3f}"
            if v != "osvd":
                c += f" ({s[f'd_{m}_vs_osvd']:+.3f}, p={s[f'p_{m}_vs_osvd']:.2g})"
            cells.append(c)
        r = f"   brain MTT r vs oSVD {s['mtt_brain_r_vs_osvd']:+.2f}" if "mtt_brain_r_vs_osvd" in s else ""
        print(f"  {v:28s} " + "  ".join(cells) + r)


if __name__ == "__main__":
    main()
