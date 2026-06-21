# C.A.S.A. Core Digital Twin (ROS 2 Humble)

This repository contains the foundational simulation infrastructure for the Compact Autonomous Surgical Arms (C.A.S.A.) project.

## 🛠️ System Requirements
* OS: Ubuntu 22.04 LTS (Native or WSL2)
* ROS 2 Distribution: Humble Hawksbill

## 🚀 Setup Instructions
Run these commands to replicate the development environment:

```bash
# 1. Install prerequisites
sudo apt update && sudo apt install git-lfs python3-vcstool -y
git lfs install

# 2. Clone this repository
git clone [https://github.com/MazenAtlam/casa-core.git](https://github.com/MazenAtlam/casa-core.git)
cd casa-core

# 3. Clone the required sub-repositories into src/
cd src
git clone [https://github.com/WPI-AIM/ambf.git](https://github.com/WPI-AIM/ambf.git)
git clone [https://github.com/collaborative-robotics/surgical_robotics_challenge.git](https://github.com/collaborative-robotics/surgical_robotics_challenge.git)

# 4. Build the workspace
cd ambf && git submodule update --init --recursive && cd ../..
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## 🧩 System Patches
The `src/casa_autonomy_stack/system_patches/` folder contains necessary modifications to third-party submodules that must be applied to the environment for this stack to work properly:
- `psm_arm.py`: ROS 2 API fixes for the PSM arm.
- `world_stereo_test.yaml`: Configuration to enable cameraL.
