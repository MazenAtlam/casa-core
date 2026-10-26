# Runbook: Quest 2 ↔ AMBF over USB and over Wi-Fi

Paths assume your repo at `~/study/graduation_project/repos/casa-core`. Every ROS terminal needs `/opt/ros/humble` and `~/ros2_ws` sourced. If `ros2` or the `quest2ros` messages aren't found in a new terminal, run:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
```

## 1. One-time setup

**1.1 Check the endpoint is on the working commit.** It must not be `4cbb3b3`:

```bash
git -C ~/ros2_ws/src/ROS-TCP-Endpoint log --oneline -1
```

Expect `90a7e4d ros2 fix`, or the PR #1 commit `1f4785d`. If it shows `4cbb3b3`, run `git -C ~/ros2_ws/src/ROS-TCP-Endpoint checkout 90a7e4d`, then `cd ~/ros2_ws && colcon build --packages-select ros_tcp_endpoint`.

**1.2 Fix the Unity scene** (open the project from Unity Hub):

First, import the TextMesh Pro essentials. They are not committed to git (see `.gitignore`), so a fresh clone doesn't have them. When the **TMP Importer** window pops up on first open, click **Import TMP Essentials**. If it doesn't pop up, use **Window → TextMeshPro → Import TMP Essential Resources**. Without them the HUD text (`Text (TMP)`) shows nothing. The scene's references reconnect automatically after the import.

1. Select `Screen → Quad`. In the Inspector, under **Mesh Renderer → Materials**, set **Size = 1**, with element 0 = `QuadUnlitTextureShader` (shader `Unlit/Texture`).
2. Select `Screen → Text (TMP)`. Set **Font Size = 3**, **Wrapping = Enabled**, **Alignment = centered**, and Rect Transform **Width 1.6, Height 0.35**.
3. Select `Bridge`. In **QuestRosBridge** assign:
   - **Screen** = `Screen`
   - **Head** = `XR Origin → Camera Offset → Main Camera`
   - **Hud** = `Text (TMP)`
   - **Video** = the `AmbfVideoQuad` component
4. In **AmbfVideoQuad**, assign **Quad** = the Quad's Mesh Renderer. Leave topic `/vr/left/image/compressed`.
5. Check for compile errors in the Console. If `ros.RosIPAddress` errors in `QuestRosBridge.cs`, delete that one HUD line.
6. Save the scene with Ctrl+S.

**1.3 Check the Unity Player settings:**

- **File → Build Settings:** platform **Android**, click **Add Open Scenes** (SampleScene must be ticked in "Scenes In Build"), tick **Development Build**.
- **Edit → Project Settings → Player → Android:** Min API 29, IL2CPP, ARM64 only, **Internet Access = Require**.

## 2. Unity Editor test (no headset, about 5 minutes)

Run this once to confirm the PC side works before touching the headset.

**Terminal 1: AMBF**
```bash
cd ~/study/graduation_project/repos/casa-core
./run_simulation_mono.sh
```
Wait for the AMBF window to show the scene.

**Terminal 2: streamer**
```bash
cd ~/study/graduation_project/repos/casa-core
python3 casa_app/casa_action/src/quest_app/ambf_jpeg_streamer.py
```
Within 5 s it logs about 15–20 fps and about 20 KB/frame. If it says "no frames received yet", AMBF isn't running. Verify with `ros2 topic list | grep ImageData`.

**Terminal 3: endpoint**
```bash
cd ~/study/graduation_project/repos/casa-core/casa_app/casa_action/src/quest_app
./start_q2r.sh
```
Expect the banner and `Starting server on 0.0.0.0:10000`.

**Unity:** **Robotics → ROS Settings**: Protocol **ROS2**, IP **127.0.0.1**, Port **10000**, Connect on Startup **on**, Show HUD **off**. Press **Play**.

**Pass:**
- The Game view shows the AMBF image on the quad.
- The HUD text shows an increasing frame count and an age of about 30 ms. `tracked: False` is expected, since there are no controllers in the Editor.
- The endpoint prints `Connection from 127.0.0.1`, then `RegisterPublisher(q2r_right_hand_pose ...) OK` and `RegisterSubscriber(/vr/left/image/compressed ...) OK`.

**Press Stop in Unity before any headset test.** Otherwise the Editor stays connected as a second client and competes for GPU and CPU.

## 3. Mode A: USB

Use this for development. It has a fixed address and the cable also charges the headset.

**3.1 Connect and check:**
```bash
adb devices
```
Expect `1WMHHA64C52473   device`. If it says `unauthorized`, put the headset on and accept "Allow USB debugging".

**3.2 Build for USB.**
1. **Robotics → ROS Settings → ROS IP address = `127.0.0.1`**.
2. **File → Build Settings → Run Device**: pick your Quest, then **Build And Run**. Save the APK as `Builds/AdasrQuestApp.apk` inside the Unity project folder (this folder is ignored by git).
3. Alternatively click **Build**, then install manually:
```bash
cd ~/study/graduation_project/repos/casa-core/casa_app/casa_action/unity/AdasrQuestApp
adb install -r Builds/AdasrQuestApp.apk
```
The first IL2CPP build takes several minutes.

**3.3 Start the PC side.** Close the app in the headset if it's running. Use the three terminals from section 2 in this order:

1. Terminal 1: `./run_simulation_mono.sh` (in the casa-core folder).
2. Terminal 2: the streamer (same command as section 2).
3. Terminal 3: `./start_q2r.sh`. The banner must say **USB headset found: adb reverse set**.

Verify the tunnel:
```bash
adb reverse --list
```
It should list a `tcp:10000` mapping. If you unplug and replug the cable, the mapping is lost: stop the endpoint with Ctrl+C and run `./start_q2r.sh` again.

**3.4 Launch the app.** Put on the headset. Pick up the **controllers** (hand tracking can make them invalid). Go to **Library → filter "Unknown Sources" → your app**. To launch from the PC instead:
```bash
adb shell pm list packages | grep -i -E "casa|adasr|quest"
adb shell monkey -p <package-from-the-list> -c android.intent.category.LAUNCHER 1
```

**3.5 Expect:**
- The endpoint prints `Connection from 127.0.0.1` followed by the Register lines.
- The HUD shows `ROS endpoint: 127.0.0.1:10000`, frames increasing, and `last` under 1 s.

## 4. Mode B: Wi-Fi

**4.1 Network.** The PC and headset must be on the same Wi-Fi network. A guest network or a hotspot with client isolation blocks it.

**4.2 Find the addresses.** The PC address is in the banner of `./start_q2r.sh` (the `LAN IP` line). To get the headset's address with it still on USB:
```bash
adb shell ip -f inet addr show wlan0 | grep inet
ping -c 3 <headset-ip>
```
Reserve the PC's IP in your router so it doesn't change.

**4.3 Firewall:**
```bash
sudo ufw status
sudo ufw allow 10000/tcp      # only if the status is "active"
```

**4.4 Build for Wi-Fi.**
1. **Robotics → ROS Settings → ROS IP address = the PC's LAN IP** (for example `192.168.1.10`).
2. Build and install as in 3.2.
3. For wireless installs, run once with USB connected:
```bash
adb tcpip 5555
adb connect <headset-ip>:5555
adb devices
```
Then unplug the cable.

**4.5 Start the PC side.** The same order as 3.3. `./start_q2r.sh` will say "No USB headset: Wi-Fi mode only" if the cable is unplugged.

**4.6 Launch the app** as in 3.4. In the endpoint terminal, expect `Connection from <headset-ip>`, and the HUD should show `ROS endpoint: <PC-ip>:10000`.

**To switch between modes:** change the IP in ROS Settings and rebuild and reinstall. Settings are saved in `Assets/Resources/ROSConnectionPrefab.prefab`, which you commit. Teammates should set their own IP after pulling.

## 5. Tests (run each in a new terminal while the app is connected)

| # | Command / action | Pass |
|---|---|---|
| T1 Topics | `ros2 topic list \| grep -E "q2r\|vr/"` | You see `/q2r_*_pose`, `_inputs`, `_thumbstick_click`, `_haptic_feedback` and `/vr/left/image/compressed` |
| T2 Pose rate | `ros2 topic hz /q2r_right_hand_pose` | About 72 Hz (the headset's refresh rate). Note the max gap |
| T3 Calibration | Hold **A+B** (right) or **X+Y** (left) for 1 s | The controller buzzes, the HUD says `calibrated: yes`, the screen recenters in front of you |
| T4 Axes | `ros2 topic echo /q2r_right_hand_pose --field pose.position`, then move 30 cm forward, left, up from the calibration pose | forward → **x +0.3**, left → **y +0.3**, up → **z +0.3** |
| T5 Inputs | `ros2 topic echo /q2r_right_hand_inputs`, press A, B, trigger, grip, move stick | The matching fields change (trigger → `press_index`, grip → `press_middle`). Note which button flips `button_upper` vs `button_lower` and write it in `quest2ros_setup.md` |
| T6 Thumbstick click | `ros2 topic echo /q2r_right_hand_thumbstick_click` | `data: true` while clicking |
| T7 Haptics | `ros2 topic pub -r 2 /q2r_right_hand_haptic_feedback quest2ros/msg/OVR2ROSHapticFeedback "{frequency: 150.0, amplitude: 1.0}"` then Ctrl+C | The right controller pulses twice a second. Use `/q2r_left_...` for the left |
| T8 Video | HUD line, plus `ros2 topic hz /vr/left/image/compressed` | Frames increasing, `last` below 1 s, rate about the streamer's fps. Write down the HUD age. It needs both clocks in sync, so treat it as approximate |
| T9 Stability | Leave the app running idle for 10 minutes | HUD still updating, no errors in the endpoint terminal |
| T10 Recording | see below | `ros2 bag info` shows messages on all four topics |

T10 commands:
```bash
mkdir -p ~/datasets/q2r_test
ros2 bag record -o ~/datasets/q2r_test/ep_$(date +%Y%m%d_%H%M%S) \
  /q2r_right_hand_pose /q2r_right_hand_inputs /q2r_left_hand_pose /q2r_left_hand_inputs
# Ctrl+C after ~10 s
ros2 bag info ~/datasets/q2r_test/ep_*
```

Run T2, T8 and T9 in both modes and compare. Fill the results into the measurements table in `quest2ros_setup.md`.

## 6. Shutdown

1. Close the app in the headset.
2. Ctrl+C in the endpoint, streamer and AMBF terminals.
3. Optional: `adb reverse --remove-all`.

If you restart the endpoint, relaunch the app. If the video stays frozen after restarting AMBF, restart the streamer.

## 7. If something fails

| Symptom | Check |
|---|---|
| App shows 0 frames, endpoint prints no `Connection from` | The wrong IP is baked in. Read the `ROS endpoint:` HUD line, fix ROS Settings, rebuild and reinstall |
| USB mode, no connection | Run `adb reverse --list`. Re-run `./start_q2r.sh` after any replug. The endpoint must be on `0.0.0.0` (the new script does it) |
| Wi-Fi mode, no connection | Check `ping <headset-ip>`, `sudo ufw status`, and that both devices are on the same non-isolated network |
| No info on the headset side | `adb logcat -c && adb logcat -s Unity` and read the connection errors. "Permission denied" points to Internet Access ≠ Require |
| Connects, drops when idle | Check the endpoint commit (1.1). If it's `90a7e4d`, try PR #1's branch, which keeps the keepalive |
| Controllers show `tracked: False` in the headset | Pick up the controllers so hand tracking turns off. Check the Oculus Touch profile in OpenXR settings |
| Pink or white quad | The Quad has more than one material (1.2) or the shader is missing |
| HUD age huge or negative | The PC and headset clocks differ by more than a few ms. Treat the number as approximate |
| Streamer fps is low | Close the Unity Editor and measure again. Unity and AMBF share the laptop's GPU and CPU |

## 8. Known issues

- Endpoint fork commit `4cbb3b3` ("Update tcp_sender.py") renamed `start_sender` to `start_`, which crashes every
  connection with `AttributeError: ... no attribute 'start_sender'`. Commit `90a7e4d` works. Upstream PR #1 (robohase)
  restores the names and keeps the new keepalive. If the Unity app drops when idle, test that PR branch.
- The store app leaves header stamps at 0. Our Unity app stamps with the headset clock.
