# DuoFusion: Physics–Learning Fusion for Extrapolating Truncated CT Perfusion

CT perfusion (CTP) selects stroke patients for thrombectomy, but scans are often stopped before the
contrast bolus washes out. DuoFusion treats a truncated scan as a short, partially observed time
series and combines two ways of extrapolating it: a physics-based perfusion twin (Windkessel flow +
transport tissue model, fitted per region and per voxel) and a learned 1D extrapolator. The
hypoperfused volume is fused on the log scale,

    log(1 + V) = w log(1 + V_twin) + (1 - w) log(1 + V_learned),

with the core from the learned extrapolator + sSVD.

The repository contains the full truncation benchmark (oSVD, sSVD, gamma-variate + oSVD/sSVD,
learned + sSVD, physics twin, hybrid, DuoFusion), the decision analysis (DEFUSE 3, DAWN), the
multiverse and conformal analyses, and the scripts of the figures.

## Main results

Median absolute error of the hypoperfused volume (ml) against the vendor Tmax > 6 s volume of the
full scan, thresholds cross-fitted on the full scans of ISLES'24 (5 patient folds) and reused
unchanged on ISLES'18.

| | ISLES'24 25 s | 30 s | 35 s | ISLES'18 25 s | 30 s | 35 s |
|---|---|---|---|---|---|---|
| oSVD | 72.4 | 50.8 | 21.3 | 7.4 | 5.8 | 3.2 |
| sSVD | 75.5 | 39.7 | 22.3 | 7.4 | 5.0 | 2.7 |
| gamma + sSVD | 64.7 | 25.4 | 20.0 | 5.8 | 2.9 | 2.2 |
| learned + sSVD | 53.3 | 25.9 | 20.8 | 3.0 | 2.8 | 2.6 |
| physics twin | **25.3** | **16.6** | **15.2** | 3.4 | 3.0 | 2.9 |
| DuoFusion (w = 0.5) | 44.8 | 23.7 | 16.5 | **2.9** | **2.4** | **2.1** |

The twin is the most accurate in-domain and flips the DEFUSE 3 decision for 19% of patients at
30 s (other estimators 26–60%); on the external cohort the learned extrapolator matches it and the
fixed 50/50 DuoFusion has the lowest median error at every truncated length (95% bootstrap CIs
overlap with the learned extrapolator). Across 90 defensible analysis pipelines, DEFUSE 3
eligibility is unstable for 13% of ISLES'24 patients and DAWN for 17%.

## Layout

```
duofusion/           package: data loading, labels, mirror registration, references, sSVD and
                     gamma-variate extrapolation (deconv.py), learned extrapolator (learned.py)
scripts/             data preparation, twin fits, truncation study, fusion, decisions, figures
external/HemoPIC_plus/  physics-based perfusion twin (vendored, see its README and LICENSE_HemoPIC)
colab/               learned extrapolator training; nnU-Net with twin maps as extra channels
paper/               figure scripts
results/             per-patient volumes and summaries of the runs reported above
tests/
```

## Data

* ISLES'24 (149 CTP scans with follow-up infarct): `hugging-science/isles24-stroke` on the
  Hugging Face Hub.
* ISLES'18 (94 CTP slabs from 63 patients, DWI core): from the ISLES 2018 challenge.

Both are public; they are not redistributed here.

## Pipeline

`DATA_DIR` (default `~/duofusion_data`) holds the data and all outputs; `PY` selects the Python
interpreter (default `python3`).

1. Prepare the cohorts in the CTP grid:
   `python scripts/prepare_isles24.py <hf_dir> $DATA_DIR/ISLES2024_Training --freesurfer <FREESURFER_HOME>`,
   `python scripts/prepare_isles18.py <TRAINING dir> $DATA_DIR/ISLES2018_Training`,
   `python scripts/add_seg4.py <data_root>`.
2. Twin fits and ROI posterior: `bash scripts/run_twin.sh 2024`, `bash scripts/run_twin.sh 2018`.
3. Truncation study (truncate, refit the twin, twin and oSVD volumes):
   `bash scripts/run_truncation.sh 25 30 35 40 45` (`YEAR=2018` for ISLES'18), and the baselines
   `bash scripts/run_baselines.sh full 25 30 35 40 45` (`METHODS="ssvd gamma_osvd gamma_ssvd"`).
4. Learned extrapolator: train with `colab/learned_extrapolation.ipynb` (one model per fold,
   `scripts/learned_folds24.json`), then
   `python scripts/osvd_volumes.py <data_root> <twin_out> <out_dir> --method learned_ssvd --models <weights>`.
5. Hybrid grids and accuracy with cross-fitted thresholds:
   `python scripts/hybrid_grids.py <results> <data_root> --yy 24 --lengths full 25 30 35 40 45`,
   `python scripts/truncation_accuracy.py <results> --yy 24`, `python scripts/weighted_fusion.py <results>`.
6. Decision fragility and conformal core volume:
   `python scripts/multiverse_eligibility.py <data_root> <out_dir>`,
   `python scripts/conformal_core.py <mv18_dir> <mv24_dir> <out_dir>`.
7. Figures: `python scripts/fig_curves_export.py ...`, `python scripts/fig_per_patient_export.py <results> fig_per_patient.json`,
   then `python paper/plot_pipeline_curves.py` and `python paper/plot_results_figs.py`.

Every script documents its inputs and outputs in its docstring.

## Requirements

Python 3.10+, `pip install -r requirements.txt`, PyTorch from https://pytorch.org (CUDA optional),
FreeSurfer (SynthSeg) for the ISLES'24 preparation, `nnunetv2` for the optional nnU-Net experiment.
