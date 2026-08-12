#!/bin/bash
# ==============================================================================
# capture_all_runs.sh — 4 orbit scan runs against the SAME running AMBF sim.
#
# The sim MUST already be running before this script starts and must stay
# running for all 4 runs. Do NOT restart it between iterations — the camera's
# live position is simulator-side state and carries over naturally.
#
# Dataset layout (2×2 factorial design):
#   run_01 — fixed light,   fixed brightness    (clean baseline / control)
#   run_02 — fixed light,   varied brightness   (isolates brightness effect)
#   run_03 — varied light,  fixed brightness    (isolates light-position effect)
#   run_04 — varied light,  varied brightness   (combined effect)
#
# Each run saves NUM_CAPTURES (500) frames → 2000 frames total.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_SCAN="$SCRIPT_DIR/orbit_scan.py"
# dataset/ goes two levels up from this script's dir (parent of parent)
BASE_OUT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/dataset/raw"

# Sanity checks
if [[ ! -f "$ORBIT_SCAN" ]]; then
    echo "[ERROR] orbit_scan.py not found at: $ORBIT_SCAN"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  capture_all_runs.sh — 2×2 factorial dataset capture"
echo "  Output root : $BASE_OUT_DIR"
echo "  Sim         : must already be running (this script will NOT start it)"
echo "======================================================================"
echo ""

# ------------------------------------------------------------------------------
# Run 01 — fixed light, fixed brightness   (control / clean baseline)
# ------------------------------------------------------------------------------
RUN="run_01"
OUT="$BASE_OUT_DIR/$RUN"
echo "=== [$RUN] fixed light | fixed brightness -> $OUT ==="
python3 "$ORBIT_SCAN" --live --capture --out-dir "$OUT"
echo "=== [$RUN] complete ==="
echo ""

# ------------------------------------------------------------------------------
# Run 02 — fixed light, varied brightness  (isolates brightness effect alone)
# ------------------------------------------------------------------------------
RUN="run_02"
OUT="$BASE_OUT_DIR/$RUN"
echo "=== [$RUN] fixed light | varied brightness -> $OUT ==="
python3 "$ORBIT_SCAN" --live --capture --vary-brightness --out-dir "$OUT"
echo "=== [$RUN] complete ==="
echo ""

# ------------------------------------------------------------------------------
# Run 03 — varied light, fixed brightness  (isolates light-position effect alone)
# ------------------------------------------------------------------------------
RUN="run_03"
OUT="$BASE_OUT_DIR/$RUN"
echo "=== [$RUN] varied light | fixed brightness -> $OUT ==="
python3 "$ORBIT_SCAN" --live --capture --vary-light --out-dir "$OUT"
echo "=== [$RUN] complete ==="
echo ""

# ------------------------------------------------------------------------------
# Run 04 — varied light, varied brightness (combined effect)
# ------------------------------------------------------------------------------
RUN="run_04"
OUT="$BASE_OUT_DIR/$RUN"
echo "=== [$RUN] varied light | varied brightness -> $OUT ==="
python3 "$ORBIT_SCAN" --live --capture --vary-light --vary-brightness --out-dir "$OUT"
echo "=== [$RUN] complete ==="
echo ""

echo "======================================================================"
echo "  All 4 runs complete — 2000 frames total"
echo "  $BASE_OUT_DIR/"
echo "    run_01/  fixed light  | fixed brightness"
echo "    run_02/  fixed light  | varied brightness"
echo "    run_03/  varied light | fixed brightness"
echo "    run_04/  varied light | varied brightness"
echo "======================================================================"
