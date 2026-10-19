#!/usr/bin/env bash
# Truncation study: for each scan length, truncate the CTP, refit HemoPIC+ (ROI fits,
# ID-CBV, ROI posterior), then DuoFusion and oSVD volumes. YEAR=2024 (default) or 2018.
#   tmux new-session -d -s trunc 'bash scripts/run_truncation.sh 30 40 > $DATA_DIR/logs/trunc.log 2>&1'
#   YEAR=2018 bash scripts/run_truncation.sh 25 30 35 40
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
HP="$CODE/external/HemoPIC_plus"
PY="${PY:-python3}"
D="${DATA_DIR:-$HOME/duofusion_data}"
W="${WORKERS:-12}"
YEAR="${YEAR:-2024}"; YY="${YEAR:2:2}"
log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -d "$D/results/osvd${YY}_full" ]; then
  log "oSVD volumes, full scans"
  "$PY" "$CODE/scripts/osvd_volumes.py" "$D/ISLES${YEAR}_Training" "$D/twin_out${YEAR}" "$D/results/osvd${YY}_full" --workers "$W" > /dev/null
fi
for L in "$@"; do
  DR="$D/ISLES${YEAR}_T$L"; OUT="$D/twin_out${YEAR}_T$L"
  log "truncate to $L s"
  "$PY" "$CODE/scripts/truncate_ctp.py" "$D/ISLES${YEAR}_Training" "$DR" --seconds "$L"
  cd "$HP"
  for c in isles24_full_aif isles24_full_aif_idvox_osvd_nc; do
    log "run_cohort $c ($L s)"
    "$PY" scripts/run_cohort.py "configs/$c.yaml" --data-root "$DR" --out-root "$OUT" --workers "$W" | tail -1
  done
  log "run_uq ($L s)"
  "$PY" scripts/run_uq.py configs/isles24_full_aif.yaml --out-root "$OUT" --workers "$W" | tail -1
  cd "$CODE"
  log "DuoFusion volumes ($L s)"
  "$PY" scripts/duofusion_volumes.py "$DR" "$OUT" "$D/results/duofusion${YY}_T$L" --draws 20 --workers "$W" > /dev/null
  log "oSVD volumes ($L s)"
  "$PY" scripts/osvd_volumes.py "$DR" "$OUT" "$D/results/osvd${YY}_T$L" --workers "$W" > /dev/null
done
log "finished"
