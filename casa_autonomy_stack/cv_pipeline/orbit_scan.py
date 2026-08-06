#!/usr/bin/env python3
"""
================================================================================
  orbit_scan.py -- Smooth full-hemisphere camera scan + dataset capture
================================================================================

MODES
-----
    python3 orbit_scan.py --discover
        Lists every object name + position. Run this first if object
        names ever change.

    python3 orbit_scan.py --test-rpy
        Holds the camera at one fixed point and cycles orientation tests,
        to visually confirm AMBF's rotation convention. (Already
        confirmed correct for this setup -- see comments in look_at_rpy.)

    python3 orbit_scan.py --live
        Watch the orbit motion with a preview window (Q to quit). Does
        NOT save anything -- just for checking the motion looks right.

    python3 orbit_scan.py --capture
        The real thing: orbits AND saves images + camera poses to disk.
        Stops automatically after NUM_CAPTURES images, or Ctrl-C/Q early
        -- either way, whatever was captured so far gets saved properly.

    python3 orbit_scan.py --capture --live
        This will capture images from the camera and save them to the 
        orbit_dataset folder along with the camera poses.
        It will also display the live feed of the camera.
 
OUTPUT (when using --capture)
------------------------------
Saved in a folder called "orbit_dataset" next to this script:
    orbit_dataset/frame_0000.png, frame_0001.png, ... 
    orbit_dataset/camera_poses.json   -- one entry per saved image, with
                                          its exact camera position/pose,
                                          needed later for mask projection.
"""

import os
import sys
import json
import time
import math
import threading
import argparse

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

try:
    from ambf_client import Client
except ImportError:
    Client = None


# ==============================================================================
# CONFIG
# ==============================================================================

CAMERA_FRAME_NAME = "CameraFrame"
PHANTOM_NAME = "Phantom"
CAMERA_TOPIC = "/ambf/env/stereo/left/ImageData"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_dataset")

ZOOM_OUT_FACTOR = 1.6     # multiplies the camera's real starting distance --
                            # 1.6 = 60% further out. Raise/lower this and
                            # re-run if it's still too close/far.

SECONDS_PER_REVOLUTION = 12.0        # fast: one full left-right spin every 12s
SECONDS_PER_ELEVATION_SWEEP = 240.0  # slow: ~20 full spins happen at each height
EL_MIN_DEG = 10.0
EL_MAX_DEG = 80.0
COMMAND_HZ = 100          # how often to send position commands (AMBF watchdog)

SPIN_SPEED_DPS = 360.0 / SECONDS_PER_REVOLUTION
EL_SPEED_DPS = (EL_MAX_DEG - EL_MIN_DEG) / SECONDS_PER_ELEVATION_SWEEP

RADIUS_VARIATION_FRACTION = 0.25   # +/- 25% zoom variation around the (already
                                      # zoomed-out) base radius. Set to 0 to disable.
SECONDS_PER_RADIUS_CYCLE = 37.0

POSITION_READ_TIMEOUT_SEC = 5.0

# --- capture settings (only used in --capture mode) ---
CAPTURE_INTERVAL_SEC = 1.0   # save one frame this often
NUM_CAPTURES = 500           # auto-stop after this many images


# ==============================================================================
# ROS2 CAMERA SUBSCRIBER
# ==============================================================================

class CamSub(Node):
    def __init__(self):
        super().__init__("orbit_scan")
        self.bridge = CvBridge()
        self.frame = None
        self._lock = threading.Lock()
        self.create_subscription(Image, CAMERA_TOPIC, self._cb, 10)

    def _cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self.frame = img.copy()
        except CvBridgeError:
            pass

    def get_frame(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

def read_position_safely(obj, label, timeout=POSITION_READ_TIMEOUT_SEC):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            p = obj.get_pos()
            candidate = (p.x, p.y, p.z)
            if any(abs(v) > 1e-6 for v in candidate):
                print(f"[OK] {label} position: "
                      f"({candidate[0]:.4f}, {candidate[1]:.4f}, {candidate[2]:.4f})")
                return candidate
        except Exception:
            pass
        time.sleep(0.1)
    print(f"[WARN] Could not get a real reading for {label} after {timeout}s.")
    return (0.0, 0.0, 0.0)


def cart_to_sph(center, pos):
    dx, dy, dz = pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    az = math.degrees(math.atan2(dx, dy))
    el = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return r, az, el


def sph_to_cart(center, r, az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    x = center[0] + r * math.cos(el) * math.sin(az)
    y = center[1] + r * math.cos(el) * math.cos(az)
    z = center[2] + r * math.sin(el)
    return (x, y, z)


def triangle_wave(value, lo, hi):
    span = hi - lo
    if span <= 0:
        return lo
    period = 2 * span
    x = (value - lo) % period
    if x > span:
        x = period - x
    return lo + x


def look_at_rpy(cam_pos, target_pos):
    """Confirmed correct via --test-rpy: at rpy=(0,0,0) the camera's local
    -Z points along world -Z (straight down), matching this derivation."""
    dx, dy, dz = target_pos[0] - cam_pos[0], target_pos[1] - cam_pos[1], target_pos[2] - cam_pos[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-9:
        return (0.0, 0.0, 0.0)
    ax, ay, az = -dx / dist, -dy / dist, -dz / dist
    pitch = math.atan2(math.sqrt(ax * ax + ay * ay), az)
    yaw = math.atan2(ay, ax) if (ax * ax + ay * ay) > 1e-9 else 0.0
    return (0.0, pitch, yaw)


def connect():
    if Client is None:
        print("[ERROR] ambf_client not importable -- run this in your AMBF Python env.")
        sys.exit(1)
    ac = Client("orbit_scan")
    ac.connect()
    time.sleep(1.5)
    return ac


# ==============================================================================
# MODE: --discover
# ==============================================================================

def run_discover():
    ac = connect()
    print("\nAll objects in the scene:")
    for name in ac.get_obj_names():
        print(f"  {name}")
    for label, name in [("Camera", CAMERA_FRAME_NAME), ("Phantom", PHANTOM_NAME)]:
        print(f"\n{label} ({name}):")
        try:
            obj = ac.get_obj_handle(name)
            time.sleep(0.5)
            read_position_safely(obj, label)
        except Exception as e:
            print(f"  could not get handle: {e}")
    ac.clean_up()


# ==============================================================================
# MODE: --test-rpy
# ==============================================================================

def run_test_rpy():
    ac = connect()
    cam = ac.get_obj_handle(CAMERA_FRAME_NAME)
    phantom = ac.get_obj_handle(PHANTOM_NAME)
    time.sleep(0.5)

    phantom_center = read_position_safely(phantom, "Phantom")
    fixed_pos = (phantom_center[0], phantom_center[1], phantom_center[2] + 0.3)

    tests = [
        ("roll=0, pitch=0, yaw=0", (0.0, 0.0, 0.0)),
        ("look_at_rpy toward phantom", look_at_rpy(fixed_pos, phantom_center)),
        ("pitch=+90deg only", (0.0, math.pi / 2, 0.0)),
        ("yaw=+90deg only", (0.0, 0.0, math.pi / 2)),
    ]

    print(f"\n[TEST] Holding camera at fixed point above phantom: {fixed_pos}")
    print("[TEST] Cycling through orientation tests every 4 seconds. Ctrl-C to stop.\n")

    try:
        while True:
            for label, rpy in tests:
                print(f"  --> commanding rpy={tuple(round(v, 3) for v in rpy)}   ({label})")
                t_end = time.time() + 4.0
                while time.time() < t_end:
                    cam.set_pos(*fixed_pos)
                    cam.set_rpy(*rpy)
                    time.sleep(1.0 / COMMAND_HZ)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl-C received.")
    ac.clean_up()


# ==============================================================================
# MODE: --live / --capture -- the actual orbit scan
# ==============================================================================

def run_scan(show_preview, capture):
    ac = connect()
    cam = ac.get_obj_handle(CAMERA_FRAME_NAME)
    phantom = ac.get_obj_handle(PHANTOM_NAME)
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print("  STARTUP DIAGNOSTICS")
    print("=" * 60)
    start_pos = read_position_safely(cam, "Camera (start)")
    phantom_center = read_position_safely(phantom, "Phantom")

    r0_measured, az0, el0 = cart_to_sph(phantom_center, start_pos)
    r0 = r0_measured * ZOOM_OUT_FACTOR
    print(f"[CHECK] Measured starting distance: {r0_measured:.4f}m")
    print(f"[CHECK] Zoomed-out orbit distance : {r0:.4f}m (x{ZOOM_OUT_FACTOR})")
    print("=" * 60 + "\n")

    el_min, el_max = EL_MIN_DEG, EL_MAX_DEG
    if el0 < el_min:
        print(f"[ADJUST] Widening EL_MIN_DEG to real starting elevation {el0:.1f}")
        el_min = el0
    if el0 > el_max:
        print(f"[ADJUST] Widening EL_MAX_DEG to real starting elevation {el0:.1f}")
        el_max = el0

    need_ros = show_preview or capture
    cam_node, spin_thread = None, None
    if need_ros:
        if not rclpy.ok():
            rclpy.init()
        cam_node = CamSub()
        spin_thread = threading.Thread(target=rclpy.spin, args=(cam_node,), daemon=True)
        spin_thread.start()
        if capture:
            print("[CAPTURE] Waiting for the camera feed to connect...")
            t0 = time.time()
            while cam_node.get_frame() is None and time.time() - t0 < 15:
                time.sleep(0.3)
            if cam_node.get_frame() is None:
                print("[ERROR] No camera frames received -- is the sim publishing images?")
                ac.clean_up()
                sys.exit(1)
            print("[CAPTURE] Camera feed connected.")

    if capture:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[CAPTURE] Saving up to {NUM_CAPTURES} images to: {OUTPUT_DIR}")
        print(f"[CAPTURE] One image every {CAPTURE_INTERVAL_SEC}s.")

    print(f"\n[SCAN] Azimuth: full revolution every {SECONDS_PER_REVOLUTION:.0f}s")
    print(f"[SCAN] Elevation: {el_min:.1f} to {el_max:.1f} deg over "
          f"{SECONDS_PER_ELEVATION_SWEEP:.0f}s (starting at {el0:.1f} deg)")
    if RADIUS_VARIATION_FRACTION > 0:
        print(f"[SCAN] Radius: {r0*(1-RADIUS_VARIATION_FRACTION):.3f}m to "
              f"{r0*(1+RADIUS_VARIATION_FRACTION):.3f}m (base {r0:.3f}m)")
    print("[SCAN] Press Ctrl-C (or Q in the preview window) to stop.\n")

    dt = 1.0 / COMMAND_HZ
    t0 = time.time()
    last_capture_t = -999.0
    saved_count = 0
    poses_log = []

    try:
        while True:
            t = time.time() - t0

            az = az0 + SPIN_SPEED_DPS * t
            el = triangle_wave(el0 + EL_SPEED_DPS * t, el_min, el_max)

            if RADIUS_VARIATION_FRACTION > 0:
                r_min = r0 * (1 - RADIUS_VARIATION_FRACTION)
                r_max = r0 * (1 + RADIUS_VARIATION_FRACTION)
                r_speed = (r_max - r_min) / (SECONDS_PER_RADIUS_CYCLE / 2)
                r = triangle_wave(r0 + r_speed * t, r_min, r_max)
            else:
                r = r0

            pos = sph_to_cart(phantom_center, r, az, el)
            roll, pitch, yaw = look_at_rpy(pos, phantom_center)
            cam.set_pos(*pos)
            cam.set_rpy(roll, pitch, yaw)

            frame = cam_node.get_frame() if cam_node is not None else None

            if capture and (t - last_capture_t) >= CAPTURE_INTERVAL_SEC:
                if frame is not None:
                    fname = f"frame_{saved_count:04d}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), frame)
                    poses_log.append({
                        "index": saved_count,
                        "image": fname,
                        "camera_pos": {"x": pos[0], "y": pos[1], "z": pos[2]},
                        "camera_rpy": {"roll": roll, "pitch": pitch, "yaw": yaw},
                        "azimuth_deg": az % 360, "elevation_deg": el, "radius_m": r,
                        "t_sec": round(t, 2),
                    })
                    saved_count += 1
                    if saved_count % 25 == 0 or saved_count == NUM_CAPTURES:
                        print(f"  [{saved_count}/{NUM_CAPTURES}] saved {fname}")
                    last_capture_t = t
                    if saved_count >= NUM_CAPTURES:
                        print(f"\n[CAPTURE] Reached target of {NUM_CAPTURES} images.")
                        break
                else:
                    print("  [!] No frame available at this moment, skipping this capture.")
                    last_capture_t = t

            if show_preview and frame is not None:
                label = f"t={t:.0f}s az={az%360:.0f}deg el={el:.0f}deg r={r:.3f}m"
                if capture:
                    label += f"  saved={saved_count}"
                cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.imshow("Orbit Scan", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[STOP] Q pressed.")
                    break

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl-C received.")
    finally:
        if capture:
            poses_path = os.path.join(OUTPUT_DIR, "camera_poses.json")
            with open(poses_path, "w") as f:
                json.dump(poses_log, f, indent=2)
            print(f"\n[CAPTURE] Saved {saved_count} images + poses to: {OUTPUT_DIR}")
            print(f"[CAPTURE] Poses file: {poses_path}")

        ac.clean_up()
        if need_ros:
            if show_preview:
                cv2.destroyAllWindows()
            cam_node.destroy_node()
            rclpy.shutdown()
            spin_thread.join(timeout=2.0)
    print("[DONE]")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--discover", action="store_true", help="List objects + positions, then exit.")
    p.add_argument("--test-rpy", action="store_true",
                    help="Hold camera fixed, cycle orientation tests.")
    p.add_argument("--live", action="store_true", help="Preview the orbit motion (no saving).")
    p.add_argument("--capture", action="store_true",
                    help="Orbit AND save images + camera_poses.json to orbit_dataset/.")
    args = p.parse_args()

    if args.discover:
        run_discover()
    elif args.test_rpy:
        run_test_rpy()
    else:
        run_scan(show_preview=args.live, capture=args.capture)