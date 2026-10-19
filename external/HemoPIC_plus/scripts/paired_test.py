"""Paired comparison of two experiments over the same patients.

python scripts/paired_test.py baseline full
python scripts/paired_test.py baseline full --out-root D:/runs

For every metric: mean difference (B - A), bootstrap 95% CI over patients and a
two-sided Wilcoxon signed-rank p-value. Lower is better for nRMSE, higher for AUC.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hemopic_plus.config import PROJECT_ROOT  # noqa: E402


def load(root: Path, name: str):
    out = defaultdict(dict)
    with open(root / name / "eval" / "per_patient.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["method"] != "model":
                continue
            key = ("nRMSE " if r["kind"] == "tracer" else "AUC ") + r["region"]
            v = float(r["value"]) if r["value"] not in ("", "nan") else np.nan
            out[key][int(r["patient"])] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs"))
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    root = Path(args.out_root)
    A, B = load(root, args.a), load(root, args.b)
    rng = np.random.default_rng(0)
    print(f"| metric | {args.a} | {args.b} | diff (B-A) | 95% CI | Wilcoxon p | n |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(A):
        pids = sorted(set(A[key]) & set(B[key]))
        a = np.array([A[key][i] for i in pids])
        b = np.array([B[key][i] for i in pids])
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if a.size < 5:
            continue
        d = b - a
        boot = d[rng.integers(0, d.size, size=(args.n_boot, d.size))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pval = wilcoxon(b, a).pvalue if np.any(d != 0) else 1.0
        print(f"| {key} | {a.mean():.4f} | {b.mean():.4f} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {pval:.2g} | {a.size} |")


if __name__ == "__main__":
    main()
