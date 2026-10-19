"""Data-driven baseline for the neural PDE: a Fourier neural operator (FNO) that maps
the arterial input to the tissue curve with no transport physics.

python scripts/train_operator.py configs/full_aif.yaml [--folds 5] [--steps 3000] [--threads 6]

Same cohort, folds (patients shuffled with seed 0, as train_npde.py) and data
term (weighted squared error over the weighted signal energy) as the neural PDE.
Each ROI gets a latent code of eight numbers, the same count as the eight
physical parameters of the neural PDE (auto-decoder): on the training patients
the network and the codes are learned together; on the held-out patients the
network is frozen and only the codes are fitted to each curve (best of
--restarts starts), as the neural PDE fits its per-ROI physics there. Inputs per
frame: the arterial curve, the time, the ROI descriptors of the closure and the
code. The curves are divided by their maximum (the data term does not depend
on the scale). Compared ROI by ROI with <out_root>/<name>__npde/eval/roi_compare.csv
when present. Writes <out_root>/<name>__operator/{summary.json, roi_compare.csv}.

--hide-blocks K also predicts hidden time frames, as scripts/eval_hidden_frames.py
does for the physics models: each of K contiguous blocks of a held-out curve is
hidden in turn, the codes are refitted on the other frames (cold: the --restarts
starts; warm: from the code fitted on all frames) and predict the hidden ones.
The curve scale is taken from the visible frames only. Error of a block: weighted
squared error on the hidden frames over the weighted signal energy of the whole
curve. Writes <out_root>/<name>__operator/hidden_frames.csv.
"""
import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde import neural as NN  # noqa: E402

LATENT = 8


class SpectralConv(torch.nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.w = torch.nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x):
        n = x.shape[-1]
        xf = torch.fft.rfft(x, n=2 * n)
        out = torch.zeros(x.shape[0], x.shape[1], xf.shape[-1], dtype=torch.cfloat)
        m = min(self.modes, xf.shape[-1])
        out[:, :, :m] = torch.einsum("bim,iom->bom", xf[:, :, :m], self.w[:, :, :m])
        return torch.fft.irfft(out, n=2 * n)[..., :n]


class FNO(torch.nn.Module):
    def __init__(self, in_ch, width=32, modes=16, layers=4):
        super().__init__()
        self.lift = torch.nn.Conv1d(in_ch, width, 1)
        self.spec = torch.nn.ModuleList(SpectralConv(width, modes) for _ in range(layers))
        self.local = torch.nn.ModuleList(torch.nn.Conv1d(width, width, 1) for _ in range(layers))
        self.proj = torch.nn.Sequential(torch.nn.Conv1d(width, 64, 1), torch.nn.GELU(), torch.nn.Conv1d(64, 1, 1))

    def forward(self, x):
        h = self.lift(x)
        for s, loc in zip(self.spec, self.local):
            h = torch.nn.functional.gelu(s(h) + loc(h))
        return self.proj(h)[:, 0]


def inputs(co, code):
    T = co.c_obs.shape[1]
    shape = (co.shape / co.shape.max(dim=1, keepdim=True).values.clamp(min=1e-12)).float()
    t = ((co.t - co.t[:, :1]) / 100.0).float()
    mask = (co.w > 0).float()
    static = torch.cat([co.z.float(), code], 1)[:, :, None].expand(-1, -1, T)
    return torch.cat([shape[:, None], t[:, None], mask[:, None], static], 1)


def data_term(co, pred):
    y = (co.c_obs / co.c_obs.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)).float()
    w = co.w.float()
    return (w * (pred - y) ** 2).sum(1) / ((w * y ** 2).sum(1) + 1e-12)


def fit(co, net, train_net, steps, lr_code=1e-2, lr_net=2e-3, l2=1e-3, seed=0, init_sd=0.0):
    torch.manual_seed(seed)
    code = (init_sd * torch.randn(co.c_obs.shape[0], LATENT)).requires_grad_(True)
    groups = [{"params": [code], "lr": lr_code}]
    if train_net:
        groups.append({"params": list(net.parameters()), "lr": lr_net})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        d = data_term(co, net(inputs(co, code)))
        loss = d.mean() + l2 * (code ** 2).sum(1).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        d = data_term(co, net(inputs(co, code)))
    return code.detach(), d


def fit_masked(co, net, w_fit, scale, steps, seed, code0=None, init_sd=0.0, lr_code=1e-2, l2=1e-3):
    """Codes fitted to the frames with w_fit > 0 (network frozen); curves divided by scale."""
    torch.manual_seed(seed)
    y = (co.c_obs / scale[:, None]).float()
    w = w_fit.float()
    den = (w * y ** 2).sum(1) + 1e-12
    code = (code0.clone() if code0 is not None else init_sd * torch.randn(co.c_obs.shape[0], LATENT)).requires_grad_(True)
    opt = torch.optim.Adam([code], lr=lr_code)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((w * (net(inputs(co, code)) - y) ** 2).sum(1) / den).mean() + l2 * (code ** 2).sum(1).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = net(inputs(co, code)) * scale[:, None].float()
        vis = (w * (pred / scale[:, None].float() - y) ** 2).sum(1) / den
    return pred.double(), vis


def hidden_rows(te, net, code_all, blocks, steps, restarts):
    B = te.c_obs.shape[0]
    rep = np.tile(np.arange(B), blocks)
    co = te.take(rep)
    blk = np.repeat(np.arange(blocks), B)
    hide = torch.zeros_like(co.w, dtype=torch.bool)
    for r in range(len(rep)):
        e = np.linspace(0, int(co.n_frames[r]), blocks + 1).astype(int)
        hide[r, e[blk[r]]:e[blk[r] + 1]] = True
    w_fit = co.w * (~hide)
    scale = (co.c_obs.abs() * (w_fit > 0)).max(dim=1).values.clamp(min=1e-12)
    energy = (co.w * co.c_obs ** 2).sum(1)
    out = {}
    starts = [("cold", dict(seed=200 + r, init_sd=0.0 if r == 0 else 0.5)) for r in range(restarts)]
    starts.append(("warm", dict(seed=0, code0=code_all[torch.as_tensor(rep)])))
    for label, kw in starts:
        pred, vis = fit_masked(co, net, w_fit, scale, steps, **kw)
        if label not in out:
            out[label] = (pred, vis)
        else:
            better = vis < out[label][1]
            out[label][0][better], out[label][1][better] = pred[better], vis[better]
    rows = []
    for r in range(len(rep)):
        row = {"subject": subject_id(int(co.subject[r])), "roi_id": int(co.roi_ids[r]), "tissue": co.tissue[r],
               "model": "operator", "block": int(blk[r]), "n_hidden": int(hide[r].sum()), "energy": float(energy[r])}
        for label in ("cold", "warm"):
            r2 = co.w[r] * (out[label][0][r] - co.c_obs[r]) ** 2
            row[f"err_hidden_{label}"] = float(r2[hide[r]].sum() / energy[r].clamp(min=1e-30))
            row[f"err_visible_{label}"] = float(r2[~hide[r]].sum() / energy[r].clamp(min=1e-30))
        rows.append(row)
    return rows


def run_fold(args):
    k, fit_dirs, train_idx, test_idx, a = args
    torch.set_num_threads(a.threads)
    t0 = time.time()
    co = NN.load_cohort(fit_dirs)
    tr, te = co.take(train_idx), co.take(test_idx)
    torch.manual_seed(k)
    net = FNO(3 + co.z.shape[1] + LATENT, a.width, a.modes, a.layers)
    _, d_tr = fit(tr, net, True, a.steps)
    for p in net.parameters():
        p.requires_grad_(False)
    best, best_code = None, None
    for r in range(a.restarts):
        code, d = fit(te, net, False, a.test_steps, seed=100 + r, init_sd=0.0 if r == 0 else 0.5)
        if best is None:
            best, best_code = d, code
        else:
            better = d < best
            best, best_code[better] = torch.where(better, d, best), code[better]
    print(f"  fold {k}: {len(train_idx)} train / {len(test_idx)} test ROIs, train data {float(d_tr.median()):.4f}, "
          f"test data {float(best.median()):.4f}, {time.time() - t0:.0f}s", flush=True)
    hrows = hidden_rows(te, net, best_code, a.hide_blocks, a.test_steps, a.restarts) if a.hide_blocks else []
    if a.hide_blocks:
        print(f"  fold {k}: hidden frames done, {time.time() - t0:.0f}s", flush=True)
    return k, best.numpy(), float(d_tr.median()), hrows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--test-steps", type=int, default=1000)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--hide-blocks", type=int, default=0, help="also predict K hidden blocks of time frames")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    root = cfg.resolve("out_root") / cfg.name
    pids = sorted(pid for pid in list_patients(root) if (root / subject_id(pid) / "fit_curves.npz").exists())
    fit_dirs = [root / subject_id(pid) for pid in pids]
    co = NN.load_cohort(fit_dirs)
    rng = np.random.default_rng(0)
    fold_of = {pid: f for pid, f in zip(rng.permutation(pids), np.arange(len(pids)) % a.folds)}
    roi_fold = np.array([fold_of[s] for s in co.subject])
    n_par = sum(p_.numel() for p_ in FNO(3 + co.z.shape[1] + LATENT, a.width, a.modes, a.layers).parameters())
    print(f"[{cfg.name}] neural operator (FNO, {n_par} weights): {len(pids)} patients, {len(co.roi_ids)} ROIs, {a.folds} folds")
    t0 = time.time()
    test = np.full(len(co.roi_ids), np.nan)
    train_med = {}
    H = []
    jobs = [(k, fit_dirs, np.where(roi_fold != k)[0], np.where(roi_fold == k)[0], a) for k in range(a.folds)]
    with ProcessPoolExecutor(max_workers=a.folds, mp_context=multiprocessing.get_context("spawn")) as ex:
        for k, d, dtr, hrows in ex.map(run_fold, jobs):
            test[roi_fold == k] = d
            train_med[k] = dtr
            H += hrows
    print(f"[{cfg.name}] all folds done in {time.time() - t0:.0f}s")
    od = root.parent / f"{cfg.name}__operator"
    od.mkdir(parents=True, exist_ok=True)
    npde = {}
    cmp_file = root.parent / f"{cfg.name}__npde" / "eval" / "roi_compare.csv"
    if cmp_file.exists():
        for r in csv.DictReader(open(cmp_file, encoding="utf8")):
            npde[(r["subject"], int(r["roi_id"]))] = r
    rows = []
    for i in range(len(co.roi_ids)):
        sub = subject_id(int(co.subject[i]))
        r = {"subject": sub, "roi_id": int(co.roi_ids[i]), "tissue": co.tissue[i], "fold": int(roi_fold[i]), "data_operator": float(test[i])}
        ref = npde.get((sub, int(co.roi_ids[i])))
        if ref is not None:
            for key in ref:
                if key.startswith("nmse"):
                    r[f"{key} (npde eval)"] = float(ref[key])
        rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("subject", "roi_id", "tissue", "fold"), k))
    with open(od / "roi_compare.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    summ = {"n_patients": len(pids), "n_roi": len(rows), "weights": n_par, "settings": vars(a) | {"config": str(a.config)},
            "train data median per fold": train_med, "held-out data median, operator": float(np.median(test))}
    for key in sorted({k for r in rows for k in r if k.endswith("(npde eval)")}):
        v = np.array([r.get(key, np.nan) for r in rows])
        ok = np.isfinite(v)
        summ[f"held-out data median, {key}"] = float(np.median(v[ok]))
        summ[f"operator better than {key}, fraction"] = float(np.mean(test[ok] < v[ok]))
    for tissue in ("GM", "WM"):
        m = np.array([r["tissue"] == tissue for r in rows])
        summ[f"held-out data median, operator, {tissue}"] = float(np.median(test[m]))
    if H:
        with open(od / "hidden_frames.csv", "w", newline="", encoding="utf8") as f:
            wr = csv.DictWriter(f, fieldnames=list(H[0].keys()))
            wr.writeheader()
            wr.writerows(H)
        for label in ("cold", "warm"):
            per_roi = {}
            for r in H:
                per_roi[(r["subject"], r["roi_id"])] = per_roi.get((r["subject"], r["roi_id"]), 0.0) + r[f"err_hidden_{label}"]
            summ[f"hidden-frame error per ROI (sum over blocks), median, {label}"] = float(np.median(list(per_roi.values())))
    (od / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf8")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
