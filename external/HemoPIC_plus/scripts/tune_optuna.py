"""Hyperparameter search with Optuna, with a held-out test set of patients.

python scripts/tune_optuna.py configs/full_aif.yaml --trials 60 --workers 30

The patients are split once (seed 0) into a tuning set (2/3) and a test set
(1/3) that the search never sees. Every trial runs the whole pipeline (and,
when chosen, the transport-PDE refit of every ROI) on the tuning patients and
returns the mean lesion ROC AUC over CBF, CBV and MTT. At the end the best
trials, the default configs and oSVD (patients with an accepted AIF) are
evaluated on the test patients, with paired Wilcoxon tests.

Writes <out_root>/optuna/{split.json, study.db, trials.csv, test.json, test_per_patient.csv}.

Nested cross-validation: --outer-folds K --outer-fold k uses fold k of a K-fold
split of the patients (seed 0) as the test set and writes to
<out_root>/optuna_cv/fold<k>/. Run k = 0..K-1, then scripts/pool_nested.py.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, evaluate_subject  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402

MAP_NAMES = [m for m, _ in MAPS]


def suggest(trial, base):
    """The search space. Returns (config, tissue model)."""
    p = {
        "k_gm": trial.suggest_int("k_gm", 4, 16),
        "k_wm": trial.suggest_int("k_wm", 3, 12),
        "coord_weight": trial.suggest_float("coord_weight", 0.0, 0.5),
        "smooth_sigma_frames": trial.suggest_float("smooth_sigma_frames", 0.8, 3.0),
        "lambda_f": trial.suggest_float("lambda_f", 0.005, 0.3, log=True),
        "lambda_v": trial.suggest_float("lambda_v", 0.005, 0.3, log=True),
        "lambda_in": trial.suggest_float("lambda_in", 0.01, 0.3, log=True),
        "tail_frac": trial.suggest_float("tail_frac", 0.1, 0.4),
        "cbf_texture": trial.suggest_categorical("cbf_texture", ["peak_ttp", "osvd"]),
        "cbf_texture_clip": trial.suggest_categorical("cbf_texture_clip", [True, False]),
        "cbv_source": trial.suggest_categorical("cbv_source", ["model_v", "id_regional", "id_voxel"]),
        "arterial_input": trial.suggest_categorical("arterial_input", ["auto_aif", "tissue_mean"]),
    }
    if p["cbf_texture_clip"]:
        p["proxy_clip_lo"] = trial.suggest_float("proxy_clip_lo", 0.1, 0.6)
        p["proxy_clip_hi"] = trial.suggest_float("proxy_clip_hi", 1.5, 5.0)
    if p["cbv_source"] != "id_voxel":
        p["cbv_ratio_clip_lo"] = trial.suggest_float("cbv_ratio_clip_lo", 0.2, 0.7)
        p["cbv_ratio_clip_hi"] = trial.suggest_float("cbv_ratio_clip_hi", 1.5, 4.0)
    p["prior_scale"] = "absolute" if p["arterial_input"] == "auto_aif" else "wm_reference"
    tissue = trial.suggest_categorical("tissue_model", ["tank", "pde"])
    return replace(base, **p), tissue


def config_from_params(base, params):
    p = {k: v for k, v in params.items() if k != "tissue_model"}
    p["prior_scale"] = "absolute" if p.get("arterial_input", base.arterial_input) == "auto_aif" else "wm_reference"
    return replace(base, **p), params.get("tissue_model", "tank")


def pde_refit(cfg, fd):
    """Replace the tank by the transport PDE for every ROI and rescale the maps."""
    import torch
    from run_pde import fit_pde, rescaled_maps
    import nibabel as nib
    from hemopic_plus.npde import transport as TP
    from hemopic_plus.uq import laplace as LP
    from hemopic_plus.uq.data import load_subject
    from hemopic_plus.uq.torch_model import derived

    torch.set_num_threads(1)
    b = load_subject(fd)
    th_pol = LP.refine_map(b)
    th, _ = fit_pde(b, th_pol, 24)
    with torch.no_grad():
        ref = {k: v.numpy() for k, v in derived(b, b.theta_fit).items()}
        new = {k: v.numpy() for k, v in derived(b, th[:, :7]).items()}
        _, c = TP.simulate(b, th, 24)
    img, cbf, cbv, mtt = rescaled_maps(fd, b.roi_ids, new["CBF"] / ref["CBF"], new["V"] / ref["V"], cfg)
    for tag, vol in (("CBF", cbf), ("CBV", cbv), ("MTT", mtt)):
        nib.save(nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header), str(fd / f"HemoPIC_{tag}.nii.gz"))
    z = dict(np.load(fd / "fit_curves.npz"))
    pos = {int(r): i for i, r in enumerate(z["roi_ids"])}
    for k, rid in enumerate(b.roi_ids):
        z["c_fit"][pos[int(rid)]] = c[k].numpy()
    np.savez_compressed(fd / "fit_curves.npz", **z)


def run_one(args):
    """Fit one patient with cfg (and the PDE if asked) and evaluate it."""
    cfg, tissue, pid, out_dir = args
    from hemopic_plus.pipeline import run_subject
    try:
        run_subject(cfg, pid, out_dir)
        if tissue == "pde":
            pde_refit(cfg, out_dir / subject_id(pid))
        rows, _ = evaluate_subject(cfg.resolve("data_root"), out_dir, pid)
    except Exception as e:
        return pid, None, repr(e)
    auc = {r["region"]: r["value"] for r in rows if r["kind"] == "roc" and r["method"] == "model"}
    nrmse = {r["region"]: r["value"] for r in rows if r["kind"] == "tracer"}
    return pid, {"auc": auc, "nrmse": nrmse}, None


def run_set(ex, cfg, tissue, pids, out_dir, retries=2):
    """Run every patient in parallel; patients that fail (e.g. out of memory on a
    shared machine) are retried one at a time in this process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res, failed = {}, []
    for pid, r, err in ex.map(run_one, [(cfg, tissue, pid, out_dir) for pid in pids]):
        if err:
            failed.append(pid)
        else:
            res[pid] = r
    for pid in failed:
        for attempt in range(retries):
            _, r, err = run_one((cfg, tissue, pid, out_dir))
            if not err:
                res[pid] = r
                break
        else:
            print(f"    patient {pid} failed after {retries} retries: {err}", flush=True)
    return res


def summary(res):
    s = {f"AUC_{m}": float(np.nanmean([r["auc"].get(m, np.nan) for r in res.values()])) for m in MAP_NAMES}
    s["AUC_mean"] = float(np.mean([s[f"AUC_{m}"] for m in MAP_NAMES]))
    s["nRMSE_brain"] = float(np.nanmean([r["nrmse"].get("full_brain", np.nan) for r in res.values()]))
    s["n"] = len(res)
    return s


def main():
    import optuna
    from scipy.stats import wilcoxon

    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--workers", type=int)
    p.add_argument("--test-frac", type=float, default=1 / 3)
    p.add_argument("--top", type=int, default=3, help="best trials evaluated on the test set")
    p.add_argument("--patients", type=int, nargs="*", help="restrict the cohort (smoke tests)")
    p.add_argument("--outer-folds", type=int, default=0, help="nested CV: number of outer folds (0 = one split)")
    p.add_argument("--outer-fold", type=int, default=0, help="nested CV: which outer fold is the test set")
    a = p.parse_args()
    base = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = base.resolve("out_root")
    od = out / "optuna" if not a.outer_folds else out / "optuna_cv" / f"fold{a.outer_fold}"
    od.mkdir(parents=True, exist_ok=True)
    pids = sorted(a.patients or list_patients(base.resolve("data_root")))
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(pids))
    if a.outer_folds:
        fold = {int(x): i % a.outer_folds for i, x in enumerate(perm)}
        test = sorted(x for x in fold if fold[x] == a.outer_fold)
        tune = sorted(x for x in fold if fold[x] != a.outer_fold)
    else:
        n_test = int(round(a.test_frac * len(pids)))
        test, tune = sorted(int(x) for x in perm[:n_test]), sorted(int(x) for x in perm[n_test:])
    (od / "split.json").write_text(json.dumps({"tune": tune, "test": test}, indent=2), encoding="utf8")
    print(f"tuning on {len(tune)} patients, {len(test)} held out for the test", flush=True)

    ex = ProcessPoolExecutor(max_workers=a.workers or os.cpu_count() or 1)
    study = optuna.create_study(study_name="hemopic_plus", direction="maximize", storage=f"sqlite:///{(od / 'study.db').as_posix()}",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True))
    if len(study.trials) == 0:
        study.enqueue_trial({"k_gm": base.k_gm, "k_wm": base.k_wm, "coord_weight": base.coord_weight,
                             "smooth_sigma_frames": base.smooth_sigma_frames, "lambda_f": base.lambda_f, "lambda_v": base.lambda_v,
                             "lambda_in": base.lambda_in, "tail_frac": base.tail_frac, "cbf_texture": "peak_ttp",
                             "cbf_texture_clip": True, "proxy_clip_lo": base.proxy_clip_lo, "proxy_clip_hi": base.proxy_clip_hi,
                             "cbv_source": "model_v", "cbv_ratio_clip_lo": base.cbv_ratio_clip_lo,
                             "cbv_ratio_clip_hi": base.cbv_ratio_clip_hi, "arterial_input": "auto_aif", "tissue_model": "tank"})
        study.enqueue_trial({"k_gm": base.k_gm, "k_wm": base.k_wm, "coord_weight": base.coord_weight,
                             "smooth_sigma_frames": base.smooth_sigma_frames, "lambda_f": base.lambda_f, "lambda_v": base.lambda_v,
                             "lambda_in": base.lambda_in, "tail_frac": base.tail_frac, "cbf_texture": "osvd",
                             "cbf_texture_clip": False, "cbv_source": "id_voxel", "arterial_input": "auto_aif", "tissue_model": "tank"})

    def objective(trial):
        t0 = time.time()
        cfg, tissue = suggest(trial, base)
        tdir = od / "trials" / f"t{trial.number:03d}"
        res = run_set(ex, cfg, tissue, tune, tdir)
        shutil.rmtree(tdir, ignore_errors=True)
        if len(res) < len(tune):
            raise optuna.TrialPruned(f"only {len(res)} of {len(tune)} patients ran")
        s = summary(res)
        for k, v in s.items():
            trial.set_user_attr(k, v)
        print(f"  trial {trial.number:3d}: mean AUC {s['AUC_mean']:.4f} (CBF {s['AUC_CBF']:.3f}, CBV {s['AUC_CBV']:.3f}, "
              f"MTT {s['AUC_MTT']:.3f}), {tissue}, {time.time() - t0:.0f}s", flush=True)
        return s["AUC_mean"]

    n_left = a.trials - len([t for t in study.trials if t.state.is_finished()])
    if n_left > 0:
        study.optimize(objective, n_trials=n_left)
    keys = sorted({k for t in study.trials for k in t.params})
    attrs = sorted({k for t in study.trials for k in t.user_attrs})
    with open(od / "trials.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["number", "state", "value", *keys, *attrs])
        for t in study.trials:
            w.writerow([t.number, t.state.name, t.value, *[t.params.get(k, "") for k in keys], *[t.user_attrs.get(k, "") for k in attrs]])

    done = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)
    cands = {f"best{i + 1} (trial {t.number})": t.params for i, t in enumerate(done[:a.top])}
    by_number = {t.number: t for t in study.trials}
    defaults = {"full_aif (default)": by_number[0].params, "full_aif_idvox_osvd_nc (default)": by_number[1].params}
    cands = {**defaults, **cands}
    test_res = {}
    for name, params in cands.items():
        cfg, tissue = config_from_params(base, params)
        tdir = od / "test" / name.split(" ")[0]
        test_res[name] = run_set(ex, cfg, tissue, test, tdir)
        shutil.rmtree(tdir, ignore_errors=True)
        print(f"  test {name:34s} " + json.dumps({k: round(v, 4) for k, v in summary(test_res[name]).items()}), flush=True)
    ex.shutdown()

    osvd = {}
    f = out / "osvd" / "roc.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf8")):
            osvd.setdefault(int(r["patient"]), {})[r["map"]] = float(r["auc"])
    report = {"split": {"tune": tune, "test": test}, "n_trials": len(done),
              "tuning_best": {"trial": done[0].number, "params": done[0].params, **done[0].user_attrs},
              "test": {}, "paired_vs_osvd": {}, "paired_vs_default": {}}
    rows = []
    for name, res in test_res.items():
        report["test"][name] = summary(res)
        for pid, r in res.items():
            rows.append({"variant": name, "patient": pid, **{f"AUC_{m}": r["auc"].get(m) for m in MAP_NAMES}})
        on_osvd = [pid for pid in res if pid in osvd]
        report["paired_vs_osvd"][name] = {}
        for m in MAP_NAMES:
            d = np.array([res[pid]["auc"][m] - osvd[pid][m] for pid in on_osvd])
            d = d[np.isfinite(d)]
            if d.size >= 5:
                report["paired_vs_osvd"][name][m] = {"n": int(d.size), "diff": float(d.mean()),
                                                     "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        ref = test_res["full_aif (default)"]
        both = [pid for pid in res if pid in ref]
        report["paired_vs_default"][name] = {}
        for m in MAP_NAMES + ["mean"]:
            if m == "mean":
                d = np.array([np.mean([res[pid]["auc"][k] for k in MAP_NAMES]) - np.mean([ref[pid]["auc"][k] for k in MAP_NAMES]) for pid in both])
            else:
                d = np.array([res[pid]["auc"][m] - ref[pid]["auc"][m] for pid in both])
            d = d[np.isfinite(d)]
            if d.size >= 5 and np.any(d != 0):
                report["paired_vs_default"][name][m] = {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue)}
    on = [pid for pid in test if pid in osvd]
    report["test"]["oSVD (accepted AIF only)"] = {f"AUC_{m}": float(np.mean([osvd[pid][m] for pid in on])) for m in MAP_NAMES} | {"n": len(on)}
    (od / "test.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    if rows:
        with open(od / "test_per_patient.csv", "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"best on the tuning set: trial {done[0].number}, mean AUC {done[0].value:.4f}")
    print("best params: " + json.dumps(done[0].params))
    for name, s in report["test"].items():
        print(f"  TEST {name:34s} CBF {s['AUC_CBF']:.3f}  CBV {s['AUC_CBV']:.3f}  MTT {s['AUC_MTT']:.3f}  n={s['n']}")
    for name, d in report["paired_vs_osvd"].items():
        print(f"  vs oSVD   {name:34s} " + "  ".join(f"{m} {v['diff']:+.3f} (p={v['p']:.2g}, n={v['n']})" for m, v in d.items()))
    for name, d in report["paired_vs_default"].items():
        print(f"  vs default {name:33s} " + "  ".join(f"{m} {v['diff']:+.3f} (p={v['p']:.2g})" for m, v in d.items()))


if __name__ == "__main__":
    main()
