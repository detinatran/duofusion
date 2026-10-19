"""One command for the whole ISLES 2017 preprocessing.

python preprocess/prepare_isles2017.py D:/data              # everything
python preprocess/prepare_isles2017.py D:/data --skip-synthseg

Steps (each is skipped when its outputs already exist):
  1. download + unzip            preprocess/download_isles2017.py
  2. flat file names             preprocess/flatten_isles2017.py
  3. per-subject TR  [C1]        preprocess/extract_tr.py  -> configs/isles2017_tr.json
  4. PWI -> concentration        preprocess/pwi_to_tracer.py (HemoPIC Preprocessing 2)
  5. SynthSeg + 4-class labels   preprocess/segmentation/*  (HemoPIC Preprocessing 1,
                                 needs FreeSurfer >= 7.3: FREESURFER_HOME, FS_LICENSE)
Then set data_root in the configs to <root>/ISLES2017_Training.
"""
import argparse
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="folder that will hold ISLES2017_Training/")
    p.add_argument("--skip-synthseg", action="store_true", help="stop before FreeSurfer (labels made elsewhere)")
    p.add_argument("--tr-out", type=Path, default=PROJECT / "configs" / "isles2017_tr.json", help="where to write the TR table")
    a = p.parse_args()
    root = a.root.expanduser().resolve()
    data = root / "ISLES2017_Training"

    run(HERE / "download_isles2017.py", root)
    run(HERE / "flatten_isles2017.py", data)
    run(HERE / "extract_tr.py", data, a.tr_out)
    run(HERE / "pwi_to_tracer.py", data, 1, 48)
    if a.skip_synthseg:
        print("\nSkipped SynthSeg. Copy hemopic_seg4/ folders from a FreeSurfer machine.")
        return
    seg_root = root / "synthseg"
    run(HERE / "segmentation" / "run_synthseg.py", data, seg_root)
    run(HERE / "segmentation" / "hemopic_seg4.py", data, seg_root)
    n = len(list(data.glob("training_*/hemopic_seg4/seg4_label_ref24.nii.gz")))
    print(f"\nDone: {n} subjects ready in {data}")


if __name__ == "__main__":
    main()
