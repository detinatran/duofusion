"""Summarise the original HemoPIC Eval/ outputs (CVT slope/r, tau medians)
for every experiment that has <out_root>/<name>/eval_original/.

Run the original scripts first, e.g. (from the HemoPIC repo):
  python Eval/cvt_check/run_eval_cvt.py DATA ../HemoPIC_plus/outputs/full ../HemoPIC_plus/outputs/full/eval_original/cvt
  python Eval/Windkessel/run_eval_tau_report.py DATA ../HemoPIC_plus/outputs/full ../HemoPIC_plus/outputs/full/eval_original/tau
then:
  python scripts/summarize_original_eval.py [--out-root outputs]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hemopic_plus.config import PROJECT_ROOT  # noqa: E402
from compare import ORDER  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs"))
    a = p.parse_args()
    root = Path(a.out_root)
    names = sorted({f.parents[4].name for f in root.glob("*/eval_original/cvt/cvt/tables/points.csv")},
                   key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
    print("| experiment | CVT slope GM / WM / Lesion / Brain | CVT r GM / WM / Lesion / Brain | mean(model MTT)/mean(Deconv MTT) Brain | tau median GM / WM / lesion / c-lesion (s) |")
    print("|---|---|---|---|---|")
    for n in names:
        pts = defaultdict(list)
        with open(root / n / "eval_original/cvt/cvt/tables/points.csv", encoding="utf8") as f:
            for r in csv.DictReader(f):
                x, y = float(r["x_data_mtt"]), float(r["y_model_ratio"])
                if np.isfinite(x) and np.isfinite(y):
                    pts[r["region"]].append((x, y))
        slopes, rs = [], []
        for reg in ("GM", "WM", "Lesion", "Brain"):
            xy = np.array(pts[reg])
            slopes.append(f"{np.polyfit(xy[:, 0], xy[:, 1], 1)[0]:.2f}")
            rs.append(f"{np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]:.2f}")
        b = np.array(pts["Brain"])
        ratio = f"{b[:, 1].mean() / b[:, 0].mean():.2f}"
        tau = "–"
        tq = root / n / "eval_original/tau/tau_quantiles_summary.csv"
        if tq.exists():
            with open(tq, encoding="utf8") as f:
                q = {r["group"]: float(r["q50"]) for r in csv.DictReader(f)}
            tau = " / ".join(f"{q.get(g, float('nan')):.1f}" for g in ("GM", "WM", "lesion", "c-lesion"))
        print(f"| {n} | {' / '.join(slopes)} | {' / '.join(rs)} | {ratio} | {tau} |")


if __name__ == "__main__":
    main()
