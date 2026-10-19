"""Side-by-side table of evaluated experiments (Markdown).

python scripts/compare.py                      # every outputs/*/eval/summary.json
python scripts/compare.py --out-root D:/runs baseline full
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hemopic_plus.config import PROJECT_ROOT  # noqa: E402

ORDER = ["baseline", "c1_tr", "c15_units", "c152_delay", "c2_delay_only", "full_ref", "c3_reffree", "full", "full_aif_ref", "full_aif_wm", "full_aif", "full_idreg", "full_idvox", "full_aif_idreg", "full_aif_idvox", "full_idvox_moment", "full_idvox_osvd", "full_aif_idvox_moment", "full_aif_idvox_osvd", "full_idvox_moment_nc", "full_idvox_osvd_nc", "full_aif_idvox_moment_nc", "full_aif_idvox_osvd_nc"]


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs"))
    p.add_argument("--save", type=Path, help="also write the table to this .md file")
    a = p.parse_args()
    root = Path(a.out_root)
    found = {f.parent.parent.name: json.loads(f.read_text(encoding="utf8")) for f in root.glob("*/eval/summary.json")}
    names = a.names or sorted(found, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
    base = found.get("baseline")

    lines = ["| experiment | N | nRMSE GM | nRMSE WM | nRMSE lesion | nRMSE c-lesion | nRMSE brain | AUC CBF | AUC CBV | AUC MTT | pooled CBF | pooled CBV | pooled MTT |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names + (["deconv (ISLES)"] if base else []):
        if n == "deconv (ISLES)":
            s, key = base, "deconv"
            tr = ["–"] * 5
        else:
            if n not in found:
                continue
            s, key = found[n], "model"
            tr = [fmt(s["tracer_nrmse"][r].get("mean")) for r in ("GM", "WM", "lesion", "c_lesion", "full_brain")]
        auc = [fmt(s["roc_auc_patient_mean"][f"{key}_{m}"].get("mean"), 4) for m in ("CBF", "CBV", "MTT")]
        pooled = [fmt(s["roc_auc_pooled"].get(f"{key}_{m}"), 4) for m in ("CBF", "CBV", "MTT")]
        lines.append(f"| {n} | {s['n_patients']} | " + " | ".join(tr + auc + pooled) + " |")
    table = "\n".join(lines)
    print(table)
    if a.save:
        a.save.write_text(table + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
