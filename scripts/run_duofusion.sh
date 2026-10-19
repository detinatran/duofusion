#!/usr/bin/env bash
# DuoFusion volumes on ISLES'24, then HemoPIC+ + DuoFusion on ISLES 2018.
#   tmux new-session -d -s duofusion 'bash scripts/run_duofusion.sh > $DATA_DIR/logs/duofusion.log 2>&1'
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
D="${DATA_DIR:-$HOME/duofusion_data}"
W="${WORKERS:-12}"
log() { echo "[$(date '+%F %T')] $*"; }
cd "$CODE"
log "DuoFusion volumes, ISLES'24"
"$PY" scripts/duofusion_volumes.py "$D/ISLES2024_Training" "$D/twin_out2024" "$D/results/duofusion24" --draws 50 --workers "$W"
log "HemoPIC+ fits, ISLES 2018"
WORKERS="$W" bash scripts/run_twin.sh 2018
log "DuoFusion volumes, ISLES 2018"
"$PY" scripts/duofusion_volumes.py "$D/ISLES2018_Training" "$D/twin_out2018" "$D/results/duofusion18" --draws 50 --workers "$W"
log "finished"
