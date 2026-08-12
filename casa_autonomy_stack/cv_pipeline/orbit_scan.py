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

    python3 orbit_scan.py --live --capture
        Orbit, show the preview window, AND save images at the same time.

    python3 orbit_scan.py --capture --out-dir <PATH>
        Save the dataset to a custom folder instead of the default
        "orbit_dataset/" next to this script. Useful for keeping separate
        fixed-light vs varied-light datasets, e.g.:
          --out-dir orbit_dataset_fixed_light
          --out-dir orbit_dataset_varied_light

    python3 orbit_scan.py --live --vary-brightness
        Watch the orbit motion while light1 also moves independently
        around the phantom. Does NOT save anything.

    python3 orbit_scan.py --capture --vary-brightness
        Orbit AND save a dataset with light1 moving independently of the
        camera, so illumination angle and viewing angle drift in and out
        of phase. Produces more diverse lighting conditions than a fixed-
        light capture.

    python3 orbit_scan.py --live --capture --vary-brightness
        Preview AND save a varied-light dataset simultaneously.

    python3 orbit_scan.py --capture --live --vary-brightness --out-dir <PATH>
        Save the varied-light dataset to a custom folder.


OUTPUT (when using --capture)
------------------------------
Saved in a folder called "orbit_dataset" next to this script:
    orbit_dataset/frame_0000.png, frame_0001.png, ...
    orbit_dataset/camera_poses.json   -- JSON object with two top-level
                                          keys:
                                            "phantom_rotation_rpy" -- the
                                              phantom's roll/pitch/yaw at
                                              scan start (constant for
                                              the whole run), needed to
                                              transform wound_faces.json
                                              from local to world space.
                                            "frames" -- one entry per
                                              saved image, with its exact
                                              camera position/pose,
                                              needed later for mask
                                              projection.

WHAT CHANGED FROM THE LAST VERSION
------------------------------------
- ZOOM_OUT_FACTOR pushes the orbit radius further out from wherever the
  camera actually starts (it was sitting too close before).
- Lighting variation removed entirely, as requested -- can revisit later.
- Image + pose capture added (--capture mode), reusing the same ROS2
  subscriber already used for the --live preview.
"""

import os
import sys
import json
import time
import math
import queue
import threading
import argparse

import cv2
import numpy as np
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
LIGHT1_NAME = "light1"   # NOT parented -- free-standing world-space light, per world_mono.yaml.
                          # Requires "lights: [light1, light2]" in world_mono.yaml + sim restart
                          # before it will show up in --discover.
CAMERA_TOPIC = "/ambf/env/stereo/left/ImageData"

OUTPUT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_dataset")

# Light1's own independent orbit -- deliberately DIFFERENT periods from the
# camera's, so illumination angle and viewing angle drift in and out of
# phase with each other over the session, instead of staying locked to a
# fixed relationship. That's what makes this genuine independent variation
# rather than just "the light follows the camera at a different offset."
LIGHT_SECONDS_PER_REVOLUTION = 47.0    # azimuth: independent (not a clean
                                          # multiple of the camera's 12s)
LIGHT_EL_MIN_DEG = 20.0
LIGHT_EL_MAX_DEG = 85.0
LIGHT_SECONDS_PER_ELEVATION_SWEEP = 133.0
LIGHT_SPIN_SPEED_DPS = 360.0 / LIGHT_SECONDS_PER_REVOLUTION
LIGHT_EL_SPEED_DPS = (LIGHT_EL_MAX_DEG - LIGHT_EL_MIN_DEG) / LIGHT_SECONDS_PER_ELEVATION_SWEEP

ZOOM_OUT_FACTOR = 1.6     # multiplies the camera's real starting distance --
                            # 1.6 = 60% further out. Raise/lower this and
                            # re-run if it's still too close/far.

SECONDS_PER_REVOLUTION = 12.0        # fast: one full left-right spin every 12s
SECONDS_PER_ELEVATION_SWEEP = 233.0  # slow: ~19.4 full spins happen at each height
EL_MIN_DEG = 10.0
EL_MAX_DEG = 80.0
COMMAND_HZ = 100          # how often to send position commands (AMBF watchdog)

SPIN_SPEED_DPS = 360.0 / SECONDS_PER_REVOLUTION
EL_SPEED_DPS = (EL_MAX_DEG - EL_MIN_DEG) / SECONDS_PER_ELEVATION_SWEEP

RADIUS_VARIATION_FRACTION = 0.25   # +/- 25% zoom variation around the (already
                                      # zoomed-out) base radius. Set to 0 to disable.
SECONDS_PER_RADIUS_CYCLE = 37.0

# Brightness variation applied directly to captured images (AMBF's light API
# only supports position/rotation, confirmed via --discover -- no intensity/
# color control exists, so this is done here instead).
BRIGHTNESS_MIN = 0.2    # 0 = black
BRIGHTNESS_MAX = 0.8    # 0.5 = unchanged, 1 = double brightness
SECONDS_PER_BRIGHTNESS_CYCLE = 71.0   # independent period, decorrelated from camera/radius

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

def read_rotation_safely(obj, label, timeout=POSITION_READ_TIMEOUT_SEC):
    """
    Mimics read_position_safely to extract the Roll, Pitch, Yaw (RPY) 
    rotation of an AMBF object.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            rpy = obj.get_rpy()
            # If the API returns a valid 3-element array, log and return it
            if rpy is not None and len(rpy) == 3:
                print(f"[OK] {label} rotation (RPY): "
                      f"({rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f})")
                return (rpy[0], rpy[1], rpy[2])
        except Exception:
            pass
        time.sleep(0.1)
        
    print(f"[WARN] Could not get a real rotation reading for {label} after {timeout}s.")
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


def compute_light_position(t, phantom_center, r0, az0, el0, el_min, el_max):
    """
    Standalone function: given elapsed time t and the light's own starting
    spherical coordinates around the phantom, returns where it should be
    right now. Uses the SAME triangle-wave technique as the camera, so it
    has the same guarantee: at t=0 this returns exactly (r0, az0, el0) --
    the light's real starting position, no jump -- and moves smoothly from
    there using ITS OWN independent speed constants (LIGHT_SPIN_SPEED_DPS /
    LIGHT_EL_SPEED_DPS), decoupled from the camera's motion.
    """
    az = az0 + LIGHT_SPIN_SPEED_DPS * t
    el = triangle_wave(el0 + LIGHT_EL_SPEED_DPS * t, el_min, el_max)
    return sph_to_cart(phantom_center, r0, az, el)



def apply_brightness(frame, intensity):
    """
    Scales image brightness. intensity=0 -> black, intensity=0.5 ->
    unchanged, intensity=1 -> double brightness (clipped at 255).
    Verified: intensity=0.2 measurably darkens, 0.8 measurably brightens,
    0.5 is a no-op, on a synthetic test frame.
    """
    factor = 2.0 * intensity
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def compute_brightness(t):
    """Smoothly varies between BRIGHTNESS_MIN and BRIGHTNESS_MAX over time,
    using the same triangle-wave technique as the camera/light position."""
    speed = (BRIGHTNESS_MAX - BRIGHTNESS_MIN) / (SECONDS_PER_BRIGHTNESS_CYCLE / 2)
    return triangle_wave(BRIGHTNESS_MIN + speed * t, BRIGHTNESS_MIN, BRIGHTNESS_MAX)


class BackgroundWriter:
    """
    Saves images on a separate thread so cv2.imwrite()'s disk I/O never
    blocks the position-commanding loop -- that blocking is the most
    likely cause of the 'Watch Dog Expired' messages, since AMBF expects
    a fresh command roughly every loop tick and a slow disk write can eat
    into that window.
    """
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            fname, frame = item
            cv2.imwrite(os.path.join(self.out_dir, fname), frame)
            self.q.task_done()

    def save(self, fname, frame):
        self.q.put((fname, frame.copy()))

    def wait_and_stop(self):
        self.q.join()          # let any queued writes finish
        self.q.put(None)
        self.thread.join(timeout=5.0)


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
    all_names = ac.get_obj_names()
    for name in all_names:
        print(f"  {name}")
    for label, name in [("Camera", CAMERA_FRAME_NAME), ("Phantom", PHANTOM_NAME)]:
        print(f"\n{label} ({name}):")
        try:
            obj = ac.get_obj_handle(name)
            time.sleep(0.5)
            read_position_safely(obj, label)
        except Exception as e:
            print(f"  could not get handle: {e}")

    # Probe whatever light objects ACTUALLY exist right now, rather than a
    # hardcoded name -- avoids the false "not found" result you'd get if a
    # light isn't active in the currently-running world config.
    light_short_names = sorted(set(
        n.split("/")[-1] for n in all_names if "light" in n.lower()
    ))
    print(f"\nFound {len(light_short_names)} active light object(s): {light_short_names}")
    for lname in light_short_names:
        print(f"\nLight '{lname}':")
        try:
            obj = ac.get_obj_handle(lname)
            time.sleep(0.5)
            read_position_safely(obj, lname)
            candidates = ["set_intensity", "get_intensity", "set_rgba", "set_rgb",
                          "set_color", "get_color", "set_spot_exponent",
                          "get_spot_exponent", "set_cutoff_angle", "get_cutoff_angle",
                          "set_attenuation", "get_attenuation"]
            found = [a for a in candidates if hasattr(obj, a)]
            all_public = [a for a in dir(obj) if not a.startswith("_")]
            print(f"  Intensity/brightness-related methods found: {found if found else 'NONE'}")
            print(f"  ALL public methods/attrs on this object: {all_public}")
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

def run_scan(show_preview, capture, out_dir, vary_light, vary_brightness):
    ac = connect()
    cam = ac.get_obj_handle(CAMERA_FRAME_NAME)
    phantom = ac.get_obj_handle(PHANTOM_NAME)
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print("  STARTUP DIAGNOSTICS")
    print("=" * 60)
    start_pos = read_position_safely(cam, "Camera (start)")
    phantom_center = read_position_safely(phantom, "Phantom")
    phantom_rotation = read_rotation_safely(phantom, "Phantom")

    light = None
    light_r0 = light_az0 = light_el0 = 0.0
    if vary_light:
        try:
            light = ac.get_obj_handle(LIGHT1_NAME)
            time.sleep(0.3)
            light_start = read_position_safely(light, f"Light '{LIGHT1_NAME}' (start)")
            light_r0, light_az0, light_el0 = cart_to_sph(phantom_center, light_start)
            print(f"[LIGHT] '{LIGHT1_NAME}' found -- will vary it independently.")
        except Exception as e:
            print(f"[ERROR] --vary-light was set but couldn't get handle for "
                  f"'{LIGHT1_NAME}': {e}")
            print("        Did you add 'lights: [light1, light2]' to world_mono.yaml "
                  "and restart the simulator? Run --discover to check.")
            sys.exit(1)

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
        os.makedirs(out_dir, exist_ok=True)
        writer = BackgroundWriter(out_dir)
        print(f"[CAPTURE] Saving up to {NUM_CAPTURES} images to: {out_dir}")
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

            light_pos = None
            if vary_light and light is not None:
                light_pos = compute_light_position(
                    t, phantom_center, light_r0, light_az0, light_el0,
                    LIGHT_EL_MIN_DEG, LIGHT_EL_MAX_DEG
                )
                light.set_pos(*light_pos)

            frame = cam_node.get_frame() if cam_node is not None else None

            if capture and (t - last_capture_t) >= CAPTURE_INTERVAL_SEC:
                if frame is not None:
                    brightness_value = None
                    save_frame = frame
                    if vary_brightness:
                        brightness_value = compute_brightness(t)
                        save_frame = apply_brightness(frame, brightness_value)

                    fname = f"frame_{saved_count:04d}.png"
                    writer.save(fname, save_frame)
                    poses_log.append({
                        "index": saved_count,
                        "image": fname,
                        "camera_pos": {"x": pos[0], "y": pos[1], "z": pos[2]},
                        "camera_rpy": {"roll": roll, "pitch": pitch, "yaw": yaw},
                        "azimuth_deg": az % 360, "elevation_deg": el, "radius_m": r,
                        "light_varied": vary_light,
                        "light_pos": ({"x": light_pos[0], "y": light_pos[1], "z": light_pos[2]}
                                       if light_pos is not None else None),
                        "brightness_varied": vary_brightness,
                        "brightness_value": brightness_value,
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
            print(f"\n[CAPTURE] Waiting for background image writer to finish "
                  f"({writer.q.qsize()} pending)...")
            writer.wait_and_stop()
            poses_path = os.path.join(out_dir, "camera_poses.json")
            with open(poses_path, "w") as f:
                json.dump({
                    "phantom_rotation_rpy": {
                        "roll": phantom_rotation[0],
                        "pitch": phantom_rotation[1],
                        "yaw": phantom_rotation[2],
                    },
                    "frames": poses_log,
                }, f, indent=2)
            print(f"[CAPTURE] Saved {saved_count} images + poses to: {out_dir}")
            print(f"[CAPTURE] Poses file: {poses_path}")
            print(f"[CAPTURE] Phantom rotation (RPY) saved: {phantom_rotation}")

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
                    help="Orbit AND save images + camera_poses.json to the output folder.")
    p.add_argument("--out-dir", default=OUTPUT_DIR_DEFAULT,
                    help="Where to save images/poses (only used with --capture). "
                         "Point this at a NEW folder for each dataset variant, "
                         "e.g. orbit_dataset_fixed_light vs orbit_dataset_varied_light.")
    p.add_argument("--vary-light", action="store_true",
                    help="Also move light1 independently while orbiting (needs "
                         "'lights: [light1, light2]' in world_mono.yaml + sim restart "
                         "first). Omit this flag entirely for a fixed-lighting dataset.")
    p.add_argument("--vary-brightness", action="store_true",
                    help="Vary image brightness (0.2-0.8 scale) directly on captured "
                         "frames -- AMBF's light API has no intensity control, so this "
                         "is applied to the images themselves instead.")
    args = p.parse_args()

    if args.discover:
        run_discover()
    elif args.test_rpy:
        run_test_rpy()
    else:
        run_scan(show_preview=args.live, capture=args.capture,
                  out_dir=args.out_dir, vary_light=args.vary_light,
                  vary_brightness=args.vary_brightness)