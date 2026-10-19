"""Download ISLES 2017 Training from Zenodo (resumable) and unzip it.

python preprocess/download_isles2017.py D:/data
-> D:/data/ISLES2017_Training.zip and D:/data/ISLES2017_Training/training_*/

The Zenodo archive holds 43 subjects (training_1..48 with 3, 17, 25, 29, 34
missing); the HemoPIC paper used 44 (it also had training_3).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


URL = "https://zenodo.org/api/records/17736412/files/ISLES2017_Training.zip/content"
SIZE = 813978491


def download(dst: Path):
    have = dst.stat().st_size if dst.exists() else 0
    if have == SIZE:
        print("zip already complete:", dst)
        return
    req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(dst, "ab" if have else "wb") as f:
        done = have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.1f} / {SIZE / 1e6:.1f} MB", end="", flush=True)
    print()
    if dst.stat().st_size != SIZE:
        raise SystemExit("Incomplete download, run again to resume")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "ISLES2017_Training.zip"
    download(zpath)
    out = root / "ISLES2017_Training"
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(out, members)
    print("extracted", len([d for d in out.glob("training_*") if d.is_dir()]), "subjects ->", out)


if __name__ == "__main__":
    main()
