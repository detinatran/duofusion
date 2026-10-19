#!/usr/bin/env bash
# HemoPIC+ fits and uncertainty on a prepared CT cohort.
#   tmux new-session -d -s twin 'bash scripts/run_twin.sh 2024 > $DATA_DIR/logs/twin24.log 2>&1'
# Arg 1: 2024 (default) or 2018. Fits go to $DATA_DIR/twin_out<year>/.
set -euo pipefail
YEAR="${1:-2024}"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
HP="$CODE/external/HemoPIC_plus"
DATA="${DATA_DIR:-$HOME/duofusion_data}/ISLES${YEAR}_Training"
OUT="${DATA_DIR:-$HOME/duofusion_data}/twin_out${YEAR}"
PY="${PY:-python3}"
W="${WORKERS:-16}"
log() { echo "[$(date '+%F %T')] $*"; }
cd "$HP"
common="--data-root $DATA --out-root $OUT"
for c in isles24_full_aif isles24_baseline isles24_full_aif_idvox_osvd_nc; do
  log "run_cohort $c"
  "$PY" scripts/run_cohort.py "configs/$c.yaml" $common --workers "$W"
done
log "eval_external (voxel fit twin-vox, oSVD with the same AIF)"
"$PY" scripts/eval_external.py $common --workers "$W" --voxel --metrics
log "run_uq (ROI posterior)"
"$PY" scripts/run_uq.py configs/isles24_full_aif.yaml --out-root "$OUT" --workers "$W"
log "finished"
