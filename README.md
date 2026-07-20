# C.A.S.A. Core Digital Twin (ROS 2 Humble)

This repository contains the foundational simulation infrastructure for the Compact Autonomous Surgical Arms (C.A.S.A.) project.

---

## 🛠️ System Requirements Matrix

Before starting, select your development platform to determine the necessary installation path:

| Platform | Recommended Setup |
| --- | --- |
| **Ubuntu 22.04 (Native)** | Recommended for best performance. |
| **Windows 10/11** | Use **WSL2** with an Ubuntu 22.04 distribution. |
| **MacOS** | Use **Docker** to run an Ubuntu 22.04 container. |

---

## 🚀 Setup Instructions

### 1. System Dependencies

Install necessary build tools and version control systems.

```bash
# Update and install essential tools
sudo apt update && sudo apt install -y \
  cmake curl software-properties-common git-lfs python3-pip python3-colcon-common-extensions
  
# Initialize Large File Storage (CRITICAL: Must be done before cloning)
git lfs install

```

### 2. Install ROS 2 Humble

*Skip this step if ROS 2 Humble is already installed.*

```bash
# Add ROS 2 repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install Desktop version
sudo apt update && sudo apt install -y ros-humble-desktop

```

### 3. Clone and Build Workspace

```bash
# Clone the main repository
git clone https://github.com/MazenAtlam/casa-core.git
cd casa-core

# Clone sub-repositories
cd casa_app
git clone https://github.com/collaborative-robotics/surgical_robotics_challenge.git

cd casa_simulation
git clone https://github.com/WPI-AIM/ambf.git

# Initialize submodules and apply system patches
cd ambf && git submodule update --init --recursive && cd ../../..
./apply_patches.sh

# Build the workspace
colcon build --symlink-install

```

### 4. Persistence Setup

Add these to the bottom of your `~/.bashrc` using any text editor (sush as `nano` or `vi`) to ensure your environment is ready on every terminal open:

```bash
source /opt/ros/humble/setup.bash
source ~/casa-core/install/setup.bash

# Graphics Overrides (For WSL2 compatibility)
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=4.6
export MESA_GLSL_VERSION_OVERRIDE=460

```

After adding these, run the following command to update your current terminal.

```bash
source ~/.bashrc

```

---

## 🕹️ Running the Simulation

Execute this command to launch the C.A.S.A. digital twin.

```bash
# Launch simulation
ambf_simulator --launch_file casa_app/surgical_robotics_challenge/launch.yaml -l 0,1,2,3,4

```

---

## ⚠️ Troubleshooting

* **Everything is white/Missing textures:** If the 3D objects still appear untextured (white), change the shader path in your launch configuration. Open `casa_app/surgical_robotics_challenge/ADF/world/world_stereo.yaml` and update the `shader_path` by replacing `rim_lighting` with `basic`.
* **Shader Compilation Errors:** If the terminal reports `invalid enumerant` or `Shader compilation failed`, ensure your OpenGL overrides are set exactly as shown in the "Running the Simulation" section.
* **Missing Arms/Actuator Errors:** Ensure you are loading indices `0,1,2,3,4` in the launch command to include both PSM1 and PSM2 arms.
