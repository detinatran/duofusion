"""ROI curve fitting.

With delay_dispersion=False and multistart=0 this is exactly HemoPIC's
fit_roi_curve (MIT License, LDR Group @ JHU): five log-parameters
(R, tau, V, F, Cin) fitted with bounded Powell against a weighted normalised
MSE plus quadratic priors. [C2] appends (delay, log dispersion) to theta;
[C4] restarts poorly fitted ROIs from random points inside the bounds.
"""
import numpy as np
from scipy.optimize import minimize

from .models import arterial_input, tracer_update, windkessel_outflow
from .signal import weight_from_aif


def fit_roi_curve(c_obs, cart_norm, t, dt, f_prior, v_prior, cin_prior, cfg, rng=None, w_curve=None):
    w = weight_from_aif(cart_norm if w_curve is None else w_curve)
    eps = 1.0 / 1e12

    t_end = float(t[t.size - 1]) if t.size >= 2 else float(dt)
    tau_min = float(max(float(dt), 0.001))
    tau_max = float(min(float(cfg.tau_cap_seconds), 0.98 * t_end))
    if not (tau_max > tau_min):
        tau_max = float(tau_min + 20.0 * float(dt))

    f0, v0, cin0 = float(f_prior), float(v_prior), float(cin_prior)
    if (not np.isfinite(f0)) or f0 <= 0.0:
        f0 = 1.0 / 1e6
    if (not np.isfinite(v0)) or v0 <= 0.0:
        v0 = 1.0 / 1e6
    if (not np.isfinite(cin0)) or cin0 <= 0.0:
        cin0 = 1.0 / 1e6

    log_r0 = float(np.log(1.0))
    log_f0, log_v0, log_cin0 = float(np.log(f0)), float(np.log(v0)), float(np.log(cin0))
    use_dd = bool(cfg.delay_dispersion)
    log_disp0 = float(np.log(max(cfg.dispersion_bounds[0], 1e-3))) if use_dd else 0.0

    def simulate(theta):
        r, tau, v, f_in, cin_bar = (float(np.exp(float(x))) for x in theta[:5])
        q_out = windkessel_outflow(t, f_in, r, tau, f_in if cfg.wk_init == "steady" else cfg.p0)
        if np.any(np.logical_not(np.isfinite(q_out))):
            return None, None
        if use_dd:
            shape = arterial_input(cart_norm, t, float(theta[5]), float(np.exp(float(theta[6]))))
        else:
            shape = cart_norm
        c_in_t = cin_bar * shape
        return q_out, tracer_update(q_out, c_in_t, v, f_in, dt, q_floor_ratio=cfg.q_floor_ratio, flow_per_minute=cfg.flow_per_minute)

    def data_mse(theta):
        q_out, c_fit = simulate(theta)
        if q_out is None:
            return 1e9, None, None
        diff = np.subtract(c_fit, c_obs)
        num = float(np.mean(np.multiply(w, np.multiply(diff, diff))))
        den = float(np.mean(np.multiply(w, np.multiply(c_obs, c_obs))) + eps)
        return float(num / den), q_out, c_fit

    def loss(theta):
        val, _, _ = data_mse(theta)
        d_r = float(np.subtract(float(theta[0]), log_r0))
        reg_r = float(cfg.reg_lambda_logr) * d_r * d_r
        d_f = float(np.subtract(float(theta[3]), log_f0))
        d_v = float(np.subtract(float(theta[2]), log_v0))
        d_cin = float(np.subtract(float(theta[4]), log_cin0))
        reg = float(cfg.lambda_f) * d_f * d_f + float(cfg.lambda_v) * d_v * d_v + float(cfg.lambda_in) * d_cin * d_cin
        if use_dd:
            d_del = float(theta[5])
            d_disp = float(theta[6]) - log_disp0
            reg += float(cfg.lambda_delay) * d_del * d_del + float(cfg.lambda_dispersion) * d_disp * d_disp
        return float(val + reg_r + reg)

    tau0 = float(max(min(0.30 * t_end, tau_max), tau_min))
    theta0 = [log_r0, np.log(tau0), log_v0, log_f0, log_cin0]

    r_lo, r_hi = 1.0 / 1e10, float(cfg.r_max)
    if r_hi <= r_lo:
        r_hi = float(r_lo * 10.0)
    tau_lo, tau_hi = float(tau_min), float(tau_max)
    if tau_hi <= tau_lo:
        tau_hi = float(tau_lo + 10.0 * float(dt))
    f_lo = float(max(f0 * 0.20, 1.0 / 1e6))
    f_hi = float(max(f0 * 5.00, f_lo * 1.10))
    v_lo = float(max(v0 * 0.20, 1.0 / 1e6))
    v_hi = float(max(v0 * 5.00, v_lo * 1.10))
    cin_lo = float(max(cin0 * 0.10, 1.0 / 1e6))
    cin_hi = float(max(cin0 * 10.0, cin_lo * 1.10))
    bounds = [
        (np.log(r_lo), np.log(r_hi)),
        (np.log(tau_lo), np.log(tau_hi)),
        (np.log(v_lo), np.log(v_hi)),
        (np.log(f_lo), np.log(f_hi)),
        (np.log(cin_lo), np.log(cin_hi)),
    ]
    if use_dd:
        theta0 += [0.0, log_disp0]
        bounds += [tuple(cfg.delay_bounds), (np.log(cfg.dispersion_bounds[0]), np.log(cfg.dispersion_bounds[1]))]
    theta0 = np.array(theta0, dtype=np.float64)
    bounds = tuple(bounds)

    res = minimize(loss, theta0, method="Powell", bounds=bounds)
    best = res
    n_starts = 1
    if int(cfg.multistart) > 0 and data_mse(res.x)[0] > float(cfg.refit_loss):
        rng = rng if rng is not None else np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        for _ in range(int(cfg.multistart)):
            start = lo + (hi - lo) * rng.uniform(0.15, 0.85, size=lo.size)
            r2 = minimize(loss, start, method="Powell", bounds=bounds)
            n_starts += 1
            if np.isfinite(r2.fun) and r2.fun < best.fun:
                best = r2

    x = best.x
    val, q_out, c_fit = data_mse(x)
    ok = bool(best.success) and (q_out is not None) and np.all(np.isfinite(q_out)) and (c_fit is not None) and np.all(np.isfinite(c_fit))
    r, tau = float(np.exp(x[0])), float(np.exp(x[1]))
    return {
        "R": r,
        "tau": tau,
        "C_wk": float(tau / r),
        "V": float(np.exp(x[2])),
        "F": float(np.exp(x[3])),
        "Cin_bar": float(np.exp(x[4])),
        "delay": float(x[5]) if use_dd else 0.0,
        "dispersion": float(np.exp(x[6])) if use_dd else 0.0,
        "loss": float(val),
        "objective": float(best.fun),
        "n_starts": int(n_starts),
        "success": bool(ok),
        "message": str(best.message),
        "Q_out": q_out.astype(np.float64) if q_out is not None else None,
        "C_fit": c_fit.astype(np.float64) if c_fit is not None else None,
    }
