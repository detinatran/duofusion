"""Forward models.

windkessel_outflow and tracer_update are the original HemoPIC models
(MIT License, LDR Group @ JHU). arterial_input adds the [C2] per-ROI bolus
delay and dispersion.
"""
import numpy as np
from scipy.signal import lfilter


def windkessel_outflow(t, q_in, r, tau, p0):
    tau = float(tau)
    if (not np.isfinite(tau)) or tau <= 0.0:
        return np.full_like(t, np.nan, dtype=np.float64)
    e = np.exp(np.divide(np.negative(t), tau))
    return q_in + np.multiply(np.subtract(float(p0) / float(r), q_in), e)


def tracer_update(q_out, c_in_t, v_const, q_in, dt, q_floor_ratio, flow_per_minute=False):
    """flow_per_minute=False is the original model: q/V is used as a rate per
    second. [C5] flow_per_minute=True treats q (ml/100g/min) and V (ml/100g)
    consistently with MTT = 60 V / F, i.e. the rate is q / (60 V) per second."""
    if flow_per_minute:
        dt = float(dt) / 60.0
    t_len = int(len(q_out))
    ct = np.zeros(t_len, dtype=np.float64)
    v = float(v_const) if np.isfinite(v_const) and v_const > 0.0 else (1.0 / 1e6)
    q_in_safe = float(q_in) if np.isfinite(q_in) and q_in >= 0.0 else 0.0
    q_floor = float(max(float(q_floor_ratio) * q_in_safe, 1.0 / 1e12))
    for n in range(t_len - 1):
        q = float(q_out[n])
        if (not np.isfinite(q)) or q < 0.0:
            q = 0.0
        q_eff = q if q > q_floor else q_floor
        kd = (q_eff / v) * float(dt)
        a = 0.0 if kd > 60.0 else float(np.exp(np.negative(kd)))
        drive = (q_in_safe / q_eff) * float(c_in_t[n]) if q_in_safe > 0.0 else 0.0
        ct[n + 1] = ct[n] * a + drive * float(np.subtract(1.0, a))
    return ct


def arterial_input(cart_norm, t, delay, dispersion):
    """[C2] Delay the shared arterial curve by `delay` seconds and convolve it with
    a unit-area exponential kernel exp(-t/dispersion)/dispersion (bolus dispersion
    model of Calamante et al.). The causal exponential is applied as an exact
    first-order recursive filter, so the cost is O(T)."""
    shifted = np.interp(t - float(delay), t, cart_norm, left=float(cart_norm[0]), right=float(cart_norm[-1]))
    if dispersion <= 0.0:
        return shifted
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    a = float(np.exp(-dt / float(dispersion)))
    out = lfilter([1.0 - a], [1.0, -a], shifted - shifted[0], zi=[0.0])[0] + shifted[0]
    return out
