"""Expose the nested Zenodo layout under the flat names HemoPIC expects.

  training_1/VSD.Brain.XX.O.MR_rCBF.127016/VSD.Brain.XX.O.MR_rCBF.127016.nii
    -> training_1/MR_rCBF.nii

Uses a symlink where the OS allows it and falls back to a copy (Windows
without developer mode cannot create symlinks).

python preprocess/flatten_isles2017.py D:/data/ISLES2017_Training
"""
import re
import shutil
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


PAT = re.compile(r"^(?:VSD|SMIR)\.Brain(?:\.XX)?\.O\.(MR_[A-Za-z0-9]+|OT)\.\d+$")


def link_or_copy(src: Path, dst: Path, subj: Path):
    try:
        dst.symlink_to(src.relative_to(subj))
    except OSError:
        shutil.copy2(src, dst)


def main():
    root = Path(sys.argv[1]).expanduser().resolve()
    n = 0
    for subj in sorted(root.glob("training_*"), key=lambda p: int(p.name.split("_")[1])):
        if not subj.is_dir():
            continue
        mods = []
        for d in sorted(subj.iterdir()):
            m = PAT.match(d.name)
            if not (d.is_dir() and m):
                continue
            src = d / (d.name + ".nii")
            if not src.exists():
                print("missing", src)
                continue
            dst = subj / (m.group(1) + ".nii")
            if not dst.exists():
                link_or_copy(src, dst, subj)
            mods.append(m.group(1))
        n += 1
        print(subj.name, " ".join(sorted(mods)))
    print("subjects", n)


if __name__ == "__main__":
    main()
