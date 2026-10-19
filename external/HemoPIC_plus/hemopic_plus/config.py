"""Experiment configuration.

Every option defaults to the original HemoPIC behaviour, so an empty config
(configs/baseline.yaml) reproduces the published pipeline bit for bit.
YAML files group keys into sections purely for readability; all keys live in
one flat namespace and unknown keys raise an error.
"""
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    name: str = "baseline"

    data_root: str = "../HemoPIC/data/ISLES2017_Training"
    out_root: str = "outputs"
    patients: Optional[List[int]] = None

    k_gm: int = 8
    k_wm: int = 6
    seed: int = 0
    min_cluster_vox: int = 400
    coord_weight: float = 0.10
    sample_max: int = 80000
    kmeans_max_iter: int = 25

    smooth_sigma_frames: float = 1.8
    baseline_frames: int = 5
    clip_nonneg: bool = True

    dt_source: str = "header"
    tr_table: str = "configs/isles2017_tr.json"
    tr_default: float = 1.44

    tau_cap_seconds: float = 300.0
    r_max: float = 5000.0
    q_floor_ratio: float = 0.02
    reg_lambda_logr: float = 0.002
    lambda_in: float = 0.06
    lambda_v: float = 0.04
    lambda_f: float = 0.04
    eps_small: float = 1.0 / 1e6
    p0: float = 1.0

    flow_per_minute: bool = False

    wk_init: str = "fixed_p0"

    delay_dispersion: bool = False
    delay_bounds: List[float] = field(default_factory=lambda: [-4.0, 10.0])
    dispersion_bounds: List[float] = field(default_factory=lambda: [0.05, 10.0])
    lambda_delay: float = 0.002
    lambda_dispersion: float = 0.002

    prior_source: str = "reference_maps"
    ref_wm_cbf: float = 22.0
    ref_wm_cbv: float = 2.0
    cbv_texture: str = "reference_maps"
    map_bounds: str = "reference_maps"
    mtt_fixed: List[float] = field(default_factory=lambda: [0.5, 24.0])
    cbv_source: str = "model_v"
    cbf_texture: str = "peak_ttp"
    moment_mtt_bounds: List[float] = field(default_factory=lambda: [0.5, 30.0])
    cbf_texture_clip: bool = True

    arterial_input: str = "tissue_mean"
    aif_auc_top: float = 0.10
    aif_rough_drop: float = 0.25
    aif_max_voxels: int = 20
    aif_smooth_sigma: float = 0.8
    aif_pve: str = "vof"
    aif_pve_max: float = 5.0
    aif_member: int = 0
    aif_pve_jitter: float = 0.15
    aif_cluster_rank: int = 0
    prior_scale: str = "wm_reference"
    k_h: float = 0.73
    rho_brain: float = 1.04

    multistart: int = 0
    refit_loss: float = 0.15

    proxy_clip_lo: float = 0.35
    proxy_clip_hi: float = 2.50
    cbv_ratio_clip_lo: float = 0.50
    cbv_ratio_clip_hi: float = 2.00
    tail_frac: float = 0.20

    workers: int = 0

    def resolve(self, key: str) -> Path:
        p = Path(getattr(self, key)).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=None) -> Config:
    cfg = Config()
    known = {f.name for f in fields(Config)}
    flat = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
        for k, v in raw.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
    flat.update(overrides or {})
    for k, v in flat.items():
        if k not in known:
            raise KeyError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    for key, allowed in (
        ("dt_source", ("header", "tr_table")),
        ("prior_source", ("reference_maps", "tracer")),
        ("cbv_texture", ("reference_maps", "tracer_auc")),
        ("map_bounds", ("reference_maps", "fixed")),
        ("arterial_input", ("tissue_mean", "auto_aif")),
        ("aif_pve", ("vof", "none")),
        ("prior_scale", ("wm_reference", "absolute")),
        ("wk_init", ("fixed_p0", "steady")),
        ("cbv_source", ("model_v", "id_regional", "id_voxel")),
        ("cbf_texture", ("peak_ttp", "moment", "osvd")),
    ):
        if getattr(cfg, key) not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {getattr(cfg, key)!r}")
    if cfg.prior_scale == "absolute" and cfg.arterial_input != "auto_aif":
        raise ValueError("prior_scale: absolute needs arterial_input: auto_aif")
    return cfg
