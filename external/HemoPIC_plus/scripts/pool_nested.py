"""Pool the outer folds of the nested cross-validation of tune_optuna.py.

python scripts/pool_nested.py [--out-root outputs]

Every patient is in exactly one outer test fold, so pooling the test-set AUCs
gives a held-out estimate on the whole cohort for: the tuned config of the
patient's fold (best trial on the other folds), the two hand-made defaults, and
oSVD (patients with an accepted AIF). Paired Wilcoxon tests; writes
<out_root>/optuna_cv/pooled.json and prints a table. Also lists the tuned
parameters of each fold, to see how stable the search is.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

MAPS = ("CBF", "CBV", "MTT")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    a = p.parse_args()
    out = Path(a.out_root)
    cv = out / "optuna_cv"
    per, params = {}, {}
    for fd in sorted(cv.glob("fold*")):
        for r in csv.DictReader(open(fd / "test_per_patient.csv", encoding="utf8")):
            v = r["variant"]
            name = "tuned" if v.startswith("best1") else v.split(" ")[0] if "default" in v else None
            if name is None:
                continue
            per.setdefault(name, {})[int(r["patient"])] = {m: float(r[f"AUC_{m}"]) for m in MAPS}
        params[fd.name] = json.loads((fd / "test.json").read_text(encoding="utf8"))["tuning_best"]["params"]
    osvd = {}
    for r in csv.DictReader(open(out / "osvd" / "roc.csv", encoding="utf8")):
        osvd.setdefault(int(r["patient"]), {})[r["map"]] = float(r["auc"])
    per["oSVD"] = osvd

    def mean_auc(d):
        return {m: float(np.nanmean([v[m] for v in d.values()])) for m in MAPS} | \
               {"mean": float(np.nanmean([np.mean([v[m] for m in MAPS]) for v in d.values()])), "n": len(d)}

    def paired(x, y, keys):
        res = {}
        for m in list(MAPS) + ["mean"]:
            d = np.array([(np.mean([x[k][q] for q in MAPS]) - np.mean([y[k][q] for q in MAPS])) if m == "mean"
                          else x[k][m] - y[k][m] for k in keys])
            d = d[np.isfinite(d)]
            res[m] = {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "n": int(d.size)}
        return res

    report = {"all_patients": {k: mean_auc(v) for k, v in per.items() if k != "oSVD"}, "params_per_fold": params,
              "paired": {}}
    common = sorted(set(per["tuned"]) & set(osvd))
    report["accepted_aif_patients"] = {k: mean_auc({p: v[p] for p in common if p in v}) for k, v in per.items()}
    for ref in ("full_aif", "full_aif_idvox_osvd_nc"):
        keys = sorted(set(per["tuned"]) & set(per[ref]))
        report["paired"][f"tuned - {ref}"] = paired(per["tuned"], per[ref], keys)
    for k in ("tuned", "full_aif", "full_aif_idvox_osvd_nc"):
        report["paired"][f"{k} - oSVD"] = paired(per[k], osvd, common)
    (cv / "pooled.json").write_text(json.dumps(report, indent=2), encoding="utf8")

    print(f"held-out lesion AUC, pooled over the outer folds ({report['all_patients']['tuned']['n']} patients)")
    for k, s in report["all_patients"].items():
        print(f"  {k:24s} CBF {s['CBF']:.3f}  CBV {s['CBV']:.3f}  MTT {s['MTT']:.3f}  mean {s['mean']:.4f}")
    print(f"on the {len(common)} patients with oSVD maps")
    for k, s in report["accepted_aif_patients"].items():
        print(f"  {k:24s} CBF {s['CBF']:.3f}  CBV {s['CBV']:.3f}  MTT {s['MTT']:.3f}  mean {s['mean']:.4f}")
    for k, d in report["paired"].items():
        print(f"  {k:34s} " + "  ".join(f"{m} {v['diff']:+.3f} (p={v['p']:.2g})" for m, v in d.items()))
    print("tuned parameters per fold:")
    for f, prm in params.items():
        print(f"  {f}: " + ", ".join(f"{k}={(round(v, 3) if isinstance(v, float) else v)}" for k, v in prm.items()))


if __name__ == "__main__":
    main()
