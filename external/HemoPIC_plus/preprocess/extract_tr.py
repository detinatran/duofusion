"""[C1] Measure the true PWI repetition time of each subject.

The NIfTI headers store pixdim[4] = 0 or 1, so HemoPIC uses dt = 1 s. The
Zenodo archive ships, next to each VSD MR_4DPWI volume, a CSV with the DICOM
Acquisition Number and Instance Creation Time of every slice. TR is the median
gap between consecutive acquisitions. SMIR subjects (36-48) have no CSV and
fall back to tr_default in the config.

python preprocess/extract_tr.py D:/data/ISLES2017_Training configs/isles2017_tr.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def tr_from_csv(path: Path):
    acq = defaultdict(list)
    with open(path, encoding="utf8", errors="replace") as f:
        rows = list(csv.reader(f))[1:]
    for r in rows:
        try:
            acq[int(r[0])].append(float(r[1]))
        except (ValueError, IndexError):
            continue
    keys = sorted(acq)
    if len(keys) < 3:
        return None, len(keys)
    starts = np.array([min(acq[k]) for k in keys])
    return float(np.median(np.diff(starts))), len(keys)


def main():
    root = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("configs/isles2017_tr.json")
    table = {}
    for subj in sorted(root.glob("training_*"), key=lambda p: int(p.name.split("_")[1])):
        pid = int(subj.name.split("_")[1])
        csvs = [p for p in subj.glob("*MR_4DPWI*/*.csv")]
        if not csvs:
            print(f"{subj.name}: no PWI metadata (uses tr_default)")
            continue
        tr, n = tr_from_csv(csvs[0])
        if tr is None:
            continue
        table[str(pid)] = round(tr, 4)
        print(f"{subj.name}: {n} frames, TR = {tr:.3f} s")
    out.write_text(json.dumps(table, indent=0), encoding="utf8")
    print(f"wrote {len(table)} subjects -> {out}")


if __name__ == "__main__":
    main()
