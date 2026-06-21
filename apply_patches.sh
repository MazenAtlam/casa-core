#!/bin/bash

echo "Applying system patches to third-party submodules..."

# Apply PSM arm patch
cp src/casa_autonomy_stack/system_patches/psm_arm.py src/surgical_robotics_challenge/scripts/surgical_robotics_challenge/psm_arm.py
echo "✅ Patched psm_arm.py"

# Apply camera config patch
cp src/casa_autonomy_stack/system_patches/world_stereo_test.yaml src/surgical_robotics_challenge/ADF/world/world_stereo_test.yaml
echo "✅ Patched world_stereo_test.yaml"

echo ""
echo "All patches applied successfully!"
echo "Please remember to rebuild the workspace if needed: colcon build --symlink-install"
