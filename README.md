# C.A.S.A. Core Digital Twin (ROS 2 Humble)

This repository contains the foundational simulation infrastructure for the Compact Autonomous Surgical Arms (C.A.S.A.) project.

## 🛠️ System Requirements Matrix

Before starting, select your development platform to determine the necessary installation path:

| Platform | Recommended Setup |
|----------|-------------------|
| **Ubuntu 22.04 (Native)** | Recommended for best performance. |
| **Windows 10/11** | Use **WSL2** with an Ubuntu 22.04 distribution. |
| **macOS** | Use **Docker** to run an Ubuntu 22.04 container. |

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

cd ../casa_simulation
git clone https://github.com/WPI-AIM/ambf.git

# Initialize submodules and apply system patches
cd ambf
git submodule update --init --recursive
cd ../../..

./apply_patches.sh

# Build the workspace
colcon build --symlink-install
```

### 4. Persistence Setup

Add these to the bottom of your `~/.bashrc` using any text editor (sush as nano or vi) to ensure your environment is ready on every terminal open:

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

## 🕹️ Running the Simulation

Execute this command to launch the C.A.S.A. digital twin.

```bash
# Launch simulation (Standard)
ambf_simulator --launch_file casa_app/surgical_robotics_challenge/launch.yaml -l 0,1,2,3,4
```

### ⚡ Fast-Paced / Optimized Simulation (Recommended)

To increase rendering speed and physics performance, use the following tuned command. This drops the physics frequency to 400 Hz and caps the communication frequency, drastically reducing CPU and network overhead for a smoother experience:

```bash
# 1. Locate the executable (if not globally available in your PATH)
EXEC_PATH=$(find ~/casa-core -name "ambf_simulator" -type f -executable | head -n 1)

# 2. Launch the optimized simulation
$EXEC_PATH --launch_file casa_app/surgical_robotics_challenge/launch.yaml \
  -l 0,1,3,4,13,14 \
  -p 400 \
  -t 1 \
  --override_max_comm_freq 60 \
  --override_min_comm_freq 30
```

## ⚙️ Performance Optimization (Single Window & Shaders)

If the simulation launches two windows (stereo vision) and runs slowly, or if you want to bypass heavy lighting shaders to increase performance, you can optimize the environment by editing the world configuration file.

Open `casa_app/surgical_robotics_challenge/ADF/world/world_stereo_test.yaml` (or `world_stereo.yaml`) and make the following changes:

### 1. Disable the Right Camera (Convert Stereo to Mono)

At the top of the file, remove `cameraR` from the `cameras` list to prevent AMBF from rendering a second window:

**Before**

```yaml
cameras: [cameraL, cameraR]
```

**After**

```yaml
cameras: [cameraL]
```

*(Optional: Scroll down to the `cameraR:` definition block later in the file and comment out the entire block using `#`.)*

### 2. Disable Heavy Shaders (If needed for performance or "white screen" bugs)

Comment out the entire `shaders:` block to fall back to basic, lightweight OpenGL shading:

**Before**

```yaml
shaders:
  path: ../../ambf_shaders/
  vertex: "shader.vs"
  fragment: "shader.fs"
```

**After**

```yaml
# shaders:
#   path: ../../ambf_shaders/
#   vertex: "shader.vs"
#   fragment: "shader.fs"
```

## ⚠️ Troubleshooting

- **Everything is white/Missing textures:** If the 3D objects still appear untextured (white), change the shader path in your launch configuration. Open `casa_app/surgical_robotics_challenge/ADF/world/world_stereo.yaml` and update the `shader_path` by replacing `rim_lighting` with `basic`. *(Alternatively, completely comment out the shaders as shown in the Performance Optimization section).*
- **Shader Compilation Errors:** If the terminal reports `invalid enumerant` or `Shader compilation failed`, ensure your OpenGL overrides are set exactly as shown in the "Running the Simulation" section.
- **Missing Arms/Actuator Errors:** Ensure you are loading indices `0,1,2,3,4` in the launch command to include both `PSM1` and `PSM2` arms.


## Landmark Jitter Dataset Generator

`landmark_jitter_augmentor.py` turns the live AMBF camera feed into a
training dataset for the imitation learning model, by taking each raw
frame and creating a realistic, randomized variation of it.

### What it does

For every frame coming in from `/ambf/env/stereo/left/ImageData`:

1. **Detect** — finds the 8 existing reddish landmark markers on the tissue.
2. **Remove** — erases them with `cv2.inpaint()` so the tissue underneath looks perfectly clean.
3. **Jitter** — picks new positions for all 8 markers with a small random shift (~8%), keeping the 4-left / 4-right formation intact.
4. **Re-draw** — pastes each marker's *real* pixels back at its new position, so it still looks like an authentic marker instead of a fake shape.
5. **Loop** — ROS2 is started once, then the script keeps grabbing frames, augmenting them, and saving them to `output/` until it has enough images.

### Output

Each generated frame produces **two files**:

```
output/frame_00000.png   ← the augmented image
output/frame_00000.json  ← the new (x, y) position of each of the 8 landmarks
```

### Why the JSON files matter

An imitation learning model doesn't just need pictures — it needs to know
*where the landmarks actually are* in each picture so it can learn to
predict that. The JSON is the ground-truth label for its matching image.
Without it, you'd have to re-detect the landmarks from the augmented
image later just to get back the coordinates the script already knew —
so they're saved right away instead.

Example:
```json
{
  "landmarks": [
    {"x": 748.9, "y": 480.6, "side": "left"},
    {"x": 824.3, "y": 484.7, "side": "right"}
  ]
}
```

### Controlling how many images are generated

At the top of the script:

```python
NUM_IMAGES_TO_GENERATE = 500
```

Change `500` to any number you want. You can also stop early at any time
with `Ctrl+C` — it will report how many images it actually saved before
exiting.