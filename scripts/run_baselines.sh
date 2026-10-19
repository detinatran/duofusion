#!/usr/bin/env bash
# sSVD and gamma-variate + oSVD volumes for the full scans and the given truncation lengths.
#   tmux new-session -d -s base 'bash scripts/run_baselines.sh full 30 40 > $DATA_DIR/logs/base.log 2>&1'
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
D="${DATA_DIR:-$HOME/duofusion_data}"
W="${WORKERS:-6}"
YEAR="${YEAR:-2024}"; YY="${YEAR:2:2}"
log() { echo "[$(date '+%F %T')] $*"; }
for L in "$@"; do
  if [ "$L" = full ]; then DR="$D/ISLES${YEAR}_Training"; OUT="$D/twin_out${YEAR}"; SUF=full
  else DR="$D/ISLES${YEAR}_T$L"; OUT="$D/twin_out${YEAR}_T$L"; SUF="T$L"; fi
  for m in ${METHODS:-ssvd gamma_osvd}; do
    [ -d "$D/results/${m}${YY}_$SUF" ] && [ "$(ls "$D/results/${m}${YY}_$SUF"/*.npz 2>/dev/null | wc -l)" -ge "${EXPECT:-90}" ] && continue
    log "$m $SUF"
    "$PY" "$CODE/scripts/osvd_volumes.py" "$DR" "$OUT" "$D/results/${m}${YY}_$SUF" --method "$m" --workers "$W" ${EXTRA:-} > /dev/null
  done
done
log "finished"
