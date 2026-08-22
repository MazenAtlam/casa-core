#!/bin/bash
# Run AMBF simulation with a single left camera and no landmarks (ghosts).
# Config files are stored in casa-core and patched into the submodule before launch.
#
# Usage: ./run_simulation_mono.sh

set -e

# Resolve the directory this script lives in (casa-core root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
SIM_DIR="$SCRIPT_DIR/casa_app/surgical_robotics_challenge"

echo "Patching submodule with mono camera config..."

# Copy world config patch (single left camera only)
cp "$SRC_DIR/casa_autonomy_stack/system_patches/world_mono.yaml" \
   "$SIM_DIR/ADF/world/world_mono.yaml"
echo "  ✅ Patched world_mono.yaml"

# Copy launch file patch (references world_mono.yaml)
cp "$SRC_DIR/casa_autonomy_stack/system_patches/launch_mono.yaml" \
   "$SIM_DIR/launch_mono.yaml"
echo "  ✅ Patched launch_mono.yaml"

echo ""
echo "Launching AMBF Simulator (one camera, no landmarks)..."
cd "$SIM_DIR"

# Locate the simulator executable, checking PATH first, then workspace install/build directories
if command -v ambf_simulator &> /dev/null; then
  SIM_PATH="ambf_simulator"
else
  # Search in workspace install/build
  SIM_PATH=$(find "$SCRIPT_DIR/install" -name "ambf_simulator" -type f -executable 2>/dev/null | head -n 1)
  if [ -z "$SIM_PATH" ]; then
    SIM_PATH=$(find "$SCRIPT_DIR/build" -name "ambf_simulator" -type f -executable 2>/dev/null | head -n 1)
  fi
  if [ -z "$SIM_PATH" ]; then
    echo "❌ Error: ambf_simulator executable not found. Make sure the workspace is built." >&2
    exit 1
  fi
fi

# -l indices: 10=Simple Phantom, 2=PSM1, 4=PSM2
# Omitted:    11=Phantom ghosts, 3=PSM1 ghosts, 5=PSM2 ghosts
$SIM_PATH --launch_file launch_mono.yaml -l 10,2,4,3,5 -p 200 -t1 \
  --override_max_comm_freq 100 --override_min_comm_freq 100
