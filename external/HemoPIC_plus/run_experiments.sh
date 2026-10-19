#!/bin/bash
# Linux/macOS: fit + evaluate every experiment, then print the comparison table.
#   bash run_experiments.sh                 # all configs
#   bash run_experiments.sh baseline full   # selected configs
set -e
cd "$(dirname "$0")"
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
NAMES=${@:-baseline c1_tr c15_units c152_delay c2_delay_only full_ref c3_reffree full full_aif_ref full_aif_wm full_aif full_idreg full_idvox full_aif_idreg full_aif_idvox full_idvox_moment full_idvox_osvd full_aif_idvox_moment full_aif_idvox_osvd}
for n in $NAMES; do
  $PY scripts/run_cohort.py configs/$n.yaml
  $PY scripts/evaluate.py configs/$n.yaml
done
$PY scripts/compare.py --save outputs/comparison.md
