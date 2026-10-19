"""Tracer transport through an ROI as a 1-D convection-dispersion PDE.

HemoPIC's tracer compartment is one well-mixed tank,

    dC/dt = k(t) * (g(t) * c_in(t) - C),   k = q_eff / (60 V) [C5],  g = F / q_eff,

whose transit-time distribution is an exponential. Here the tank is replaced by
a tube s in [0, 1] (s = fraction of the ROI blood volume already passed) with
plug flow and longitudinal dispersion D = 1 / Pe:

    dc/dt = k(t) * ( -dc/ds + D d2c/ds2 ),
    inlet  (Danckwerts)  c - D dc/ds = g(t) c_in(t),
    outlet               dc/ds = 0,
    C(t) = integral of c over s.

Integrating over s gives dC/dt = k (g c_in - c(1)): mass is conserved and the
mean transit time is 1/k, so CBF, CBV and MTT = 60 V / CBF keep their meaning
(central volume theorem). Pe -> 0 is HemoPIC's well-mixed tank, Pe -> inf is
plug flow; in between the transit times follow the first-passage distribution
of a drift-diffusion process, which is narrower than the exponential.

Discretisation: N finite volumes, upwind convection, central dispersion. N = 1
is exactly HemoPIC's tank for every Pe. Within a frame k and g c_in are frozen
(as in tracer_update) and the linear system is advanced exactly. The generator
is tridiagonal and similar to a symmetric matrix, so the tissue curve is a sum
of N exponential modes; one symmetric eigendecomposition per ROI replaces a
matrix exponential per frame (advance_expm is kept as the reference).

The arterial input keeps HemoPIC+'s per-ROI delay and exponential dispersion
kernel [C2]; log Pe is an eighth parameter. With Pe -> 0 the model is therefore
exactly HemoPIC+ (tank + input kernel), so the PDE fit can only improve on it.

Parallel paths: the ROI can be split into M tubes that share the inlet
concentration, with blood-volume fractions nu_m and flow fractions phi_m
(each summing to 1). Tube m runs at rate k phi_m / nu_m and the tissue curve is
sum_m nu_m C_m. The mean transit time stays sum_m phi_m (nu_m / phi_m) / k = 1/k,
so MTT = 60 V / CBF is unchanged, but the spread of transit times (capillary
transit-time heterogeneity) can exceed that of a single tube. These fractions are
what the neural closure in npde/neural.py predicts. Unequal cell volumes and
dispersion multipliers along one tube (vol, dmul) are supported as well; they
also keep mass and the mean transit time.
"""
import numpy as np
import torch

from ..uq.torch_model import DTYPE, interp_shift

NAMES = ["log_R", "log_tau", "log_V", "log_F", "log_Cin", "delay", "log_disp", "log_Pe"]


def tridiag(log_pe, n, vol=None, dmul=None):
    """Tridiagonal generator A (per unit k) of dc/dt = k (A c + b u), b = e_0 / vol_0:
    returns the diagonal (B, n), upper and lower diagonals (B, n-1) and vol."""
    B = log_pe.shape[0]
    if vol is None:
        vol = torch.full((B, n), 1.0 / n, dtype=DTYPE)
    if n == 1:
        e = torch.zeros(B, 0, dtype=DTYPE)
        return -1.0 / vol, e, e, vol
    D = torch.exp(-log_pe)[:, None]
    dface = D * (1.0 if dmul is None else dmul) / (0.5 * (vol[:, 1:] + vol[:, :-1]))
    pad = torch.zeros(B, 1, dtype=DTYPE)
    diag = -(1.0 + torch.cat([pad, dface], 1) + torch.cat([dface, pad], 1)) / vol
    up = dface / vol[:, :-1]
    low = (1.0 + dface) / vol[:, 1:]
    return diag, up, low, vol


def operator(log_pe, n, vol=None, dmul=None):
    """Dense augmented generator [[A, b], [0, 0]] (B, n+1, n+1), for the
    matrix-exponential reference solver."""
    diag, up, low, vol = tridiag(log_pe, n, vol, dmul)
    B = diag.shape[0]
    G = torch.zeros(B, n + 1, n + 1, dtype=DTYPE)
    idx = torch.arange(n)
    G[:, idx, idx] = diag
    G[:, idx[:-1], idx[1:]] = up
    G[:, idx[1:], idx[:-1]] = low
    G[:, 0, n] = 1.0 / vol[:, 0]
    return G, vol


def advance_expm(kdt, u, G, vol):
    """Reference solver: one augmented matrix exponential per frame (slow)."""
    B, T = kdt.shape
    n = vol.shape[1]
    M = torch.linalg.matrix_exp(kdt[:, :-1, None, None] * G[:, None])
    Phi, gam = M[..., :n, :n], M[..., :n, n]
    c = torch.zeros(B, n, dtype=DTYPE)
    out = [torch.zeros(B, dtype=DTYPE)]
    for k in range(T - 1):
        c = (Phi[:, k] @ c[..., None])[..., 0] + gam[:, k] * u[:, k, None]
        out.append((vol * c).sum(dim=1))
    return torch.stack(out, dim=1)


def modes(diag, up, low, vol):
    """Eigen-modes of A. A tridiagonal matrix with positive off-diagonal
    products is similar to a symmetric one (A = S^-1 Abar S), so its eigenvalues
    lam are real and distinct, and the tissue curve is a sum of exponential modes:
    C = sum_j y_j, y_j' = k (lam_j y_j + wgt_j u), wgt_j = gamma_j beta_j."""
    B, n = diag.shape
    if n == 1:
        return diag, torch.ones(B, 1, dtype=DTYPE) / vol[:, :1]
    log_up, log_low = torch.log(up.clamp(min=1e-300)), torch.log(low)
    log_s = torch.cat([torch.zeros(B, 1, dtype=DTYPE), torch.cumsum(0.5 * (log_up - log_low), dim=1)], dim=1)
    off = torch.exp(0.5 * (log_up + log_low))
    Abar = torch.diag_embed(diag) + torch.diag_embed(off, 1) + torch.diag_embed(off, -1)
    lam, Q = torch.linalg.eigh(Abar)
    beta = Q[:, 0, :] / vol[:, :1]
    gamma = (vol[:, :, None] * torch.exp(-log_s)[:, :, None] * Q).sum(1)
    return lam, gamma * beta


def transit_density(log_pe, n, s, vol=None, dmul=None):
    """Transit-time density h(s) of the tube (outflow after a unit inlet impulse),
    s in units of the mean transit time 1/k. The tank (n = 1) gives exp(-s)."""
    diag, up, low, vol = tridiag(log_pe, n, vol, dmul)
    B = diag.shape[0]
    A = torch.diag_embed(diag)
    if n > 1:
        A = A + torch.diag_embed(up, 1) + torch.diag_embed(low, -1)
    b = torch.zeros(B, n, 1, dtype=DTYPE)
    b[:, 0, 0] = 1.0 / vol[:, 0]
    out = []
    for si in s:
        c = torch.linalg.matrix_exp(float(si) * A) @ b
        out.append(c[:, -1, 0])
    return torch.stack(out, dim=1)


def transit_cv(log_pe, n, paths=None):
    """Coefficient of variation of the transit times (CTH / MTT). Moments of the
    outflow density h(t) = e_n^T exp(A t) b: m_k = k! e_n^T (-A)^-(k+1) b. With
    parallel tubes T = X / r_m with probability phi_m, r_m = phi_m / nu_m, so
    E[T^2] = (1 + cv_tube^2) sum_m nu_m^2 / phi_m. The tank has cv = 1."""
    diag, up, low, vol = tridiag(log_pe, n)
    A = torch.diag_embed(diag)
    if n > 1:
        A = A + torch.diag_embed(up, 1) + torch.diag_embed(low, -1)
    b = torch.zeros(A.shape[0], n, 1, dtype=DTYPE)
    b[:, 0, 0] = 1.0 / vol[:, 0]
    x1 = torch.linalg.solve(-A, b)
    x2 = torch.linalg.solve(-A, x1)
    x3 = torch.linalg.solve(-A, x2)
    m0, m1, m2 = x1[:, -1, 0], x2[:, -1, 0], 2.0 * x3[:, -1, 0]
    cv2 = m2 * m0 / m1 ** 2 - 1.0
    if paths is not None:
        nu, phi = paths
        cv2 = (1.0 + cv2) * (nu ** 2 / phi).sum(1) - 1.0
    return torch.sqrt(cv2.clamp(min=0.0))


def transit_density_paths(log_pe, n, s, nu, phi):
    """Transit-time density of parallel tubes: sum_m phi_m r_m h(r_m s), r_m = phi_m / nu_m."""
    out = 0.0
    for m in range(nu.shape[1]):
        r = (phi[:, m] / nu[:, m]).detach().numpy()
        h = torch.stack([transit_density(log_pe[i:i + 1], n, np.asarray(s) * r[i])[0] for i in range(len(r))])
        out = out + phi[:, m, None] * torch.as_tensor(r, dtype=DTYPE)[:, None] * h
    return out


def advance(kdt, u, diag, up, low, vol):
    """Exact advance with k and u frozen within each frame, in modal form."""
    lam, wgt = modes(diag, up, low, vol)
    B, T = kdt.shape
    y = torch.zeros_like(lam)
    out = [torch.zeros(B, dtype=DTYPE)]
    for k in range(T - 1):
        e = torch.exp(kdt[:, k, None] * lam)
        y = e * y + wgt * (e - 1.0) / lam * u[:, k, None]
        out.append(y.sum(dim=1))
    return torch.stack(out, dim=1)


def disperse_rows(x, dt, tau_d):
    """[C2] unit-area exponential kernel exp(-t/tau_d)/tau_d applied as the exact
    recursive filter of models.arterial_input (on x - x[0]); dt scalar or (B,)."""
    a = torch.exp(-torch.as_tensor(dt, dtype=DTYPE) / tau_d)
    base = x[:, :1]
    u = x - base
    ys = [torch.zeros_like(u[:, 0])]
    for n in range(1, u.shape[1]):
        ys.append(a * ys[-1] + (1.0 - a) * u[:, n])
    return torch.stack(ys, dim=1) + base


def drive(cfg, t, dt, shape_in, theta):
    """Windkessel outflow, k dt and inlet concentration u = (F / q) Cin shape_in."""
    r, tau, v, f, cin = (torch.exp(theta[:, k]) for k in range(5))
    p0 = f if cfg.wk_init == "steady" else torch.full_like(f, float(cfg.p0))
    q_out = f[:, None] + (p0 / r - f)[:, None] * torch.exp(-t / tau[:, None])
    dt_eff = dt / 60.0 if cfg.flow_per_minute else dt
    q_eff = torch.maximum(torch.clamp(q_out, min=0.0), torch.clamp(float(cfg.q_floor_ratio) * f, min=1e-12)[:, None])
    kdt = (q_eff / v[:, None]) * dt_eff
    u = (f[:, None] / q_eff) * cin[:, None] * shape_in
    return q_out, kdt, u


def tissue(kdt, u, log_pe, n_cells, paths=None, vol=None, dmul=None):
    """Tissue curve of one tube, or of parallel tubes paths = (nu, phi), each (B, M)."""
    if paths is None:
        return advance(kdt, u, *tridiag(log_pe, int(n_cells), vol, dmul))
    nu, phi = paths
    B, M = nu.shape
    kk = (kdt[:, None, :] * (phi / nu)[:, :, None]).reshape(B * M, -1)
    uu = u[:, None, :].expand(B, M, u.shape[1]).reshape(B * M, -1)
    lp = log_pe[:, None].expand(B, M).reshape(-1)
    c = advance(kk, uu, *tridiag(lp, int(n_cells))).reshape(B, M, -1)
    return (nu[:, :, None] * c).sum(1)


def simulate(batch, theta, n_cells=24, paths=None, vol=None, dmul=None):
    """theta (B, 8): log R, log tau, log V, log F, log Cin, delay, log dispersion, log Pe.
    Returns q_out (B, T) and the tissue curve C (B, T)."""
    shape_in = disperse_rows(interp_shift(batch.shape, batch.t, theta[:, 5]), batch.dt, torch.exp(theta[:, 6]))
    q_out, kdt, u = drive(batch.cfg, batch.t[None, :], batch.dt, shape_in, theta)
    return q_out, tissue(kdt, u, theta[:, 7], n_cells, paths, vol, dmul)


def bounds(batch, pe_bounds=(0.02, 100.0)):
    """HemoPIC+ bounds (7 parameters) plus log Pe."""
    B = batch.lo.shape[0]
    lo = torch.cat([batch.lo, torch.full((B, 1), float(np.log(pe_bounds[0])), dtype=DTYPE)], 1)
    hi = torch.cat([batch.hi, torch.full((B, 1), float(np.log(pe_bounds[1])), dtype=DTYPE)], 1)
    return lo, hi


def priors(cfg, theta, log_f0, log_v0, log_cin0):
    """The quadratic log-space priors of fit_roi_curve ([C2] included); none on log Pe."""
    log_disp0 = float(np.log(max(cfg.dispersion_bounds[0], 1e-3)))
    return (float(cfg.reg_lambda_logr) * theta[:, 0] ** 2
            + float(cfg.lambda_f) * (theta[:, 3] - log_f0) ** 2
            + float(cfg.lambda_v) * (theta[:, 2] - log_v0) ** 2
            + float(cfg.lambda_in) * (theta[:, 4] - log_cin0) ** 2
            + float(cfg.lambda_delay) * theta[:, 5] ** 2
            + float(cfg.lambda_dispersion) * (theta[:, 6] - log_disp0) ** 2)


def objective(batch, theta, n_cells=24, paths=None, w=None):
    """HemoPIC's objective: weighted normalised MSE + the same priors."""
    _, c_fit = simulate(batch, theta, n_cells, paths)
    ww = batch.w if w is None else w
    num = (ww * (c_fit - batch.c_obs) ** 2).mean(dim=1)
    den = (ww * batch.c_obs ** 2).mean(dim=1) + 1e-12
    data = num / den
    return data + priors(batch.cfg, theta, batch.log_f0, batch.log_v0, batch.log_cin0), data, num, den
