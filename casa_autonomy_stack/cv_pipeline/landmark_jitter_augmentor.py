#!/usr/bin/env python3
"""
================================================================================
  landmark_jitter_augmentor.py
  CASA Surgical Robotics Challenge — Landmark Jitter Dataset Generator
================================================================================

PURPOSE
-------
Turns the live AMBF camera feed into a training-data generator for an
imitation learning model. For every incoming frame it:

    1. DETECT   Finds the 8 existing dark/reddish landmark markers on the
                (bright) tissue.
    2. REMOVE   Uses cv2.inpaint() to erase the originals so the tissue
                looks perfectly clean underneath.
    3. JITTER   Computes new (x, y) positions for all 8 markers with a
                small random offset (~5-10% of the natural marker
                spacing), keeping the 4-left / 4-right formation intact.
    4. RE-DRAW  Pastes the markers' EXACT ORIGINAL PIXELS at their new
                jittered locations (see MARKER_DRAW_MODE note below).
    5. LOOP     ROS2 is started ONCE. Frames are pulled from a fast local
                loop and saved to output/ until N images are collected.

WHY "PASTE ORIGINAL PIXELS" INSTEAD OF cv2.circle()?
-----------------------------------------------------
MARKER_DRAW_MODE = "realistic" (default) copies the real marker's pixels
(dark reddish diamond, soft shading, perspective) from their old location
and blends them at the new one. This keeps every augmented image
photorealistic — the model never sees a "fake-looking" marker that
doesn't match what the real simulator/camera actually produces.

If you ever want quick, obvious, high-contrast dots instead (e.g. for a
fast sanity-check dataset), flip MARKER_DRAW_MODE to "circle". Both paths
go through the same draw_landmark() dispatcher, so switching is a
one-line change — see the CONFIG section below.

OUTPUT
------
    output/frame_00000.png, frame_00001.png, ...
    output/frame_00000.json, ...   (new landmark (x, y) labels, optional)

HOW TO RUN
----------
    Make sure the AMBF simulator is already running, then:
        source ~/ros2_ws/install/setup.bash
        python3 landmark_jitter_augmentor.py

DEPENDENCIES
------------
    pip install opencv-python-headless numpy
    sudo apt install ros-humble-cv-bridge

CAMERA TOPIC INFO
------------------
    Topic     : /ambf/env/stereo/left/ImageData
    Encoding  : bgr8 (converted by CvBridge)
    Resolution: 640 x 480 pixels
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import json
import time
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


# ==============================================================================
# CONFIGURATION — edit these to change behavior
# ==============================================================================

CAMERA_TOPIC = "/ambf/env/stereo/left/ImageData"
IMG_WIDTH = 640
IMG_HEIGHT = 480
NUM_LANDMARKS = 8  # 4 markers on each side of the wound

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
NUM_IMAGES_TO_GENERATE = 500  # how many augmented frames to save before stopping
SAVE_LABELS = True  # also dump a .json with each image's new landmark (x, y) coords

# --- STEP 1: detection thresholds -------------------------------------------
# Markers are dark/reddish; tissue background is bright and low-saturation.
# These HSV ranges were sampled directly off a real AMBF frame:
#   marker pixels  ~ H=1,   S=76-145, V=195-207   (reddish)
#   tissue pixels  ~ H=0,   S=0,      V=255       (white, no saturation)
#   grey backdrop  ~ H=20,  S=5,      V=168       (also no saturation)
# So "hue in the red band AND saturation above a floor" isolates the
# markers cleanly from both backgrounds without needing a brightness cutoff.
HSV_RED_RANGES = [
    {"lower": np.array([0, 30, 0]), "upper": np.array([15, 255, 255])},
    {"lower": np.array([160, 30, 0]), "upper": np.array([179, 255, 255])},
]
MIN_MARKER_AREA = 40     # px^2 — filters out stray noise specks
MAX_MARKER_AREA = 5000   # px^2 — filters out any oversized false positive
MORPH_KERNEL_SIZE = 3

# --- STEP 2: inpainting (removal) -------------------------------------------
INPAINT_RADIUS = 4
INPAINT_METHOD = cv2.INPAINT_TELEA          # cv2.INPAINT_NS also works well
INPAINT_MASK_DILATE_PX = 5                  # grows the erase-mask a little so
INPAINT_MASK_DILATE_ITERS = 2               # soft/anti-aliased edges vanish too

# --- STEP 3: jitter ----------------------------------------------------------
JITTER_PERCENT = 0.08        # 8% of natural marker spacing (within your 5-10% ask)
MIN_CENTERLINE_MARGIN = 8    # px — keeps left markers left / right markers right

# --- STEP 4: re-draw ---------------------------------------------------------
MARKER_DRAW_MODE = "realistic"   # "realistic" (paste original pixels) or "circle"
PATCH_PADDING = 4                # px of context captured around each marker
FEATHER_BLUR_KSIZE = 3           # softens the pasted patch's edge so it blends
CIRCLE_COLOR = (0, 0, 200)       # BGR — only used when MARKER_DRAW_MODE == "circle"
CIRCLE_RADIUS = 8                # only used when MARKER_DRAW_MODE == "circle"

SHOW_WINDOW = False  # live preview while generating (needs a display)


# ==============================================================================
# STEP 1 — ROS2 CAMERA SUBSCRIBER
# ==============================================================================

class AmbfCameraSubscriber(Node):
    """
    Subscribes once to the AMBF left camera topic and keeps the most recent
    frame (plus its ROS timestamp) available for the main loop to pull from.
    rclpy.spin() runs in a background thread — started once in main() — so
    this node just needs to store whatever arrives.
    """

    def __init__(self):
        super().__init__("landmark_jitter_augmentor")
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_stamp = None
        self._lock = threading.Lock()

        self.subscription = self.create_subscription(
            msg_type=Image,
            topic=CAMERA_TOPIC,
            callback=self.image_callback,
            qos_profile=10,
        )
        self.get_logger().info(f"Subscribed to camera topic: {CAMERA_TOPIC}")

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            with self._lock:
                self.latest_frame = cv_image.copy()
                self.latest_stamp = stamp
        except CvBridgeError as e:
            self.get_logger().error(f"[CvBridge Error] {e}")

    def get_frame_and_stamp(self):
        """Thread-safe fetch of (frame, stamp). Returns (None, None) if
        nothing has arrived yet."""
        with self._lock:
            if self.latest_frame is None:
                return None, None
            return self.latest_frame.copy(), self.latest_stamp


# ==============================================================================
# STEP 1 — LANDMARK DETECTOR
# ==============================================================================

class LandmarkDetector:
    """
    Finds the 8 dark/reddish landmark markers on the bright tissue and
    returns everything downstream steps need: each marker's centroid,
    padded bounding box, a local binary mask of its exact shape, and a
    cropped pixel patch (used later to paste the marker back at its new
    jittered position).
    """

    def __init__(self):
        self.hsv_ranges = HSV_RED_RANGES
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )

    def detect(self, bgr_frame: np.ndarray):
        """
        Returns:
            (landmarks, full_mask, ok)
            landmarks : list of dicts (len == NUM_LANDMARKS) or None
            full_mask : uint8 (H, W) binary mask of all detected markers,
                        used later for inpainting, or None
            ok        : False if fewer than NUM_LANDMARKS markers were found
                        this frame (caller should fall back / skip)
        """
        H, W = bgr_frame.shape[:2]
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        mask = np.zeros((H, W), dtype=np.uint8)
        for r in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r["lower"], r["upper"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours
                    if MIN_MARKER_AREA <= cv2.contourArea(c) <= MAX_MARKER_AREA]

        if len(contours) < NUM_LANDMARKS:
            return None, None, False

        # Keep the NUM_LANDMARKS largest blobs — drops any stray false positive.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:NUM_LANDMARKS]

        landmarks = []
        for c in contours:
            M = cv2.moments(c)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(c)

            x0, y0 = max(x - PATCH_PADDING, 0), max(y - PATCH_PADDING, 0)
            x1, y1 = min(x + w + PATCH_PADDING, W), min(y + h + PATCH_PADDING, H)

            local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.drawContours(local_mask, [c], -1, 255, thickness=-1, offset=(-x0, -y0))

            landmarks.append({
                "center": (cx, cy),
                "bbox": (x0, y0, x1, y1),
                "local_mask": local_mask,
                "patch": bgr_frame[y0:y1, x0:x1].copy(),
            })

        return landmarks, mask, True


# ==============================================================================
# STEP 2 — REMOVAL (INPAINT)
# ==============================================================================

def remove_landmarks(frame: np.ndarray, detection_mask: np.ndarray) -> np.ndarray:
    """
    Erases the original markers with cv2.inpaint() so the tissue underneath
    looks perfectly clean. The mask is dilated slightly first — skip that
    and the soft, anti-aliased edge pixels around each marker get left
    behind as a faint reddish halo.
    """
    dilate_kernel = np.ones((INPAINT_MASK_DILATE_PX, INPAINT_MASK_DILATE_PX), np.uint8)
    dilated_mask = cv2.dilate(detection_mask, dilate_kernel, iterations=INPAINT_MASK_DILATE_ITERS)
    return cv2.inpaint(frame, dilated_mask, INPAINT_RADIUS, INPAINT_METHOD)


# ==============================================================================
# STEP 3 — JITTER
# ==============================================================================

def compute_new_positions(landmarks: list) -> list:
    """
    Computes a new (x, y) for every landmark: a small random offset
    (JITTER_PERCENT of the natural marker spacing) around its original
    position, while guaranteeing the 4-left / 4-right formation survives —
    no marker is allowed to jitter across the wound's centerline.

    Mutates each landmark dict in place to add a "side" key ("left"/"right").
    Returns a list of (new_x, new_y) tuples, same order as `landmarks`.
    """
    xs = [lm["center"][0] for lm in landmarks]
    centerline = float(np.median(xs))
    for lm in landmarks:
        lm["side"] = "left" if lm["center"][0] < centerline else "right"

    def side_spacing(side):
        ys = sorted(lm["center"][1] for lm in landmarks if lm["side"] == side)
        diffs = np.diff(ys)
        return float(np.median(diffs)) if len(diffs) else 60.0

    ref_scale = float(np.mean([side_spacing("left"), side_spacing("right")]))

    rng = np.random.default_rng()
    new_positions = []
    for lm in landmarks:
        cx, cy = lm["center"]
        jx = rng.uniform(-1, 1) * JITTER_PERCENT * ref_scale
        jy = rng.uniform(-1, 1) * JITTER_PERCENT * ref_scale
        nx, ny = cx + jx, cy + jy

        # Keep the formation intact: clamp back across the centerline if needed.
        if lm["side"] == "left":
            nx = min(nx, centerline - MIN_CENTERLINE_MARGIN)
        else:
            nx = max(nx, centerline + MIN_CENTERLINE_MARGIN)

        new_positions.append((nx, ny))
    return new_positions


def _clamp_to_frame(cx, cy, w, h, img_w, img_h):
    """Keeps a marker's bounding box fully inside the image after jitter."""
    half_w, half_h = w / 2.0, h / 2.0
    cx = float(np.clip(cx, half_w, img_w - half_w))
    cy = float(np.clip(cy, half_h, img_h - half_h))
    return cx, cy


# ==============================================================================
# STEP 4 — RE-DRAW
# ==============================================================================
# Two interchangeable drawing modes behind one dispatcher (draw_landmark).
# Change MARKER_DRAW_MODE at the top of the file to switch between them —
# nothing else in the pipeline needs to change.
# ==============================================================================

def draw_landmark_realistic(canvas: np.ndarray, landmark: dict, new_center: tuple) -> None:
    """
    Pastes the marker's ORIGINAL pixels at its new jittered location using
    its exact contour shape as an alpha mask (feathered slightly so the
    paste blends instead of leaving a hard rectangular edge).
    """
    x0, y0, x1, y1 = landmark["bbox"]
    w, h = x1 - x0, y1 - y0
    old_cx, old_cy = landmark["center"]

    dx = int(round(new_center[0] - old_cx))
    dy = int(round(new_center[1] - old_cy))
    nx0, ny0 = x0 + dx, y0 + dy
    nx1, ny1 = nx0 + w, ny0 + h

    H, W = canvas.shape[:2]
    if nx0 < 0 or ny0 < 0 or nx1 > W or ny1 > H:
        # Shouldn't normally trigger (compute_new_positions + _clamp_to_frame
        # already keep markers well inside the frame) — guard anyway.
        nx0 = int(np.clip(nx0, 0, W - w))
        ny0 = int(np.clip(ny0, 0, H - h))
        nx1, ny1 = nx0 + w, ny0 + h

    alpha = cv2.GaussianBlur(
        landmark["local_mask"].astype(np.float32) / 255.0,
        (FEATHER_BLUR_KSIZE, FEATHER_BLUR_KSIZE), 0,
    )[..., None]

    region = canvas[ny0:ny1, nx0:nx1].astype(np.float32)
    patch = landmark["patch"].astype(np.float32)
    canvas[ny0:ny1, nx0:nx1] = (region * (1 - alpha) + patch * alpha).astype(np.uint8)


def draw_landmark_circle(canvas: np.ndarray, landmark: dict, new_center: tuple) -> None:
    """
    Simple, high-contrast solid-circle marker. Not used by default — flip
    MARKER_DRAW_MODE to "circle" if you want this instead of realistic
    pasted markers (e.g. for a quick sanity-check dataset).
    """
    x, y = int(round(new_center[0])), int(round(new_center[1]))
    cv2.circle(canvas, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, thickness=-1)
    cv2.circle(canvas, (x, y), CIRCLE_RADIUS, (255, 255, 255), thickness=1)  # outline


def draw_landmark(canvas: np.ndarray, landmark: dict, new_center: tuple,
                   mode: str = MARKER_DRAW_MODE) -> None:
    if mode == "realistic":
        draw_landmark_realistic(canvas, landmark, new_center)
    elif mode == "circle":
        draw_landmark_circle(canvas, landmark, new_center)
    else:
        raise ValueError(f"Unknown MARKER_DRAW_MODE: {mode!r}")


# ==============================================================================
# ORCHESTRATOR — ties DETECT -> REMOVE -> JITTER -> RE-DRAW into one call
# ==============================================================================

class LandmarkAugmentor:
    """
    Runs the full per-frame augmentation pipeline. Caches the last
    successful detection so a single noisy/occluded frame (e.g. motion
    blur, a tool briefly covering a marker) doesn't stall dataset
    generation — it just re-uses the last known-good marker positions
    and patches for that one frame instead of skipping it outright.
    """

    def __init__(self):
        self.detector = LandmarkDetector()
        self._last_good_landmarks = None
        self._last_good_mask = None

    def augment(self, frame: np.ndarray):
        """
        Returns (augmented_image, labels_dict, status_message).
        augmented_image is None if augmentation could not be performed
        at all (no cached fallback available yet).
        """
        landmarks, mask, ok = self.detector.detect(frame)

        if ok:
            self._last_good_landmarks = landmarks
            self._last_good_mask = mask
            status = "ok"
        elif self._last_good_landmarks is not None:
            landmarks, mask = self._last_good_landmarks, self._last_good_mask
            status = "used cached landmark positions (detection missed this frame)"
        else:
            return None, None, "no valid detection yet - skipping frame"

        H, W = frame.shape[:2]
        canvas = remove_landmarks(frame, mask)
        new_positions = compute_new_positions(landmarks)

        final_positions = []
        for lm, (nx, ny) in zip(landmarks, new_positions):
            x0, y0, x1, y1 = lm["bbox"]
            nx, ny = _clamp_to_frame(nx, ny, x1 - x0, y1 - y0, W, H)
            draw_landmark(canvas, lm, (nx, ny), MARKER_DRAW_MODE)
            final_positions.append((nx, ny, lm["side"]))

        labels = {
            "landmarks": [
                {"x": round(x, 1), "y": round(y, 1), "side": side}
                for x, y, side in final_positions
            ]
        }
        return canvas, labels, status


# ==============================================================================
# MAIN — starts ROS2 ONCE, then loops locally to generate the dataset
# ==============================================================================

def main():
    print("=" * 70)
    print("  CASA — Landmark Jitter Dataset Generator")
    print("=" * 70)

    # --- Start ROS2 exactly once ---
    print("\n[STEP 1] Initializing ROS2...")
    rclpy.init()
    camera_node = AmbfCameraSubscriber()
    spin_thread = threading.Thread(target=rclpy.spin, args=(camera_node,), daemon=True)
    spin_thread.start()
    print(f"[STEP 1] Subscribed to {CAMERA_TOPIC}. Waiting for the first frame...")

    # --- Wait for the first frame ---
    timeout_seconds = 30
    frame, stamp = None, None
    start_time = time.time()
    while frame is None:
        frame, stamp = camera_node.get_frame_and_stamp()
        if frame is None:
            if time.time() - start_time > timeout_seconds:
                print(f"[ERROR] No frame received after {timeout_seconds}s. "
                      f"Is the AMBF simulator running?")
                rclpy.shutdown()
                sys.exit(1)
            time.sleep(0.5)
    print(f"[STEP 1] First frame received! Shape: {frame.shape}")

    # --- Prepare output ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    augmentor = LandmarkAugmentor()

    # --- Fast local loop: no re-init of ROS2, just pull + process + save ---
    print(f"\n[LOOP] Generating {NUM_IMAGES_TO_GENERATE} augmented frames "
          f"-> {OUTPUT_DIR}")
    saved = 0
    last_stamp = None
    try:
        while saved < NUM_IMAGES_TO_GENERATE:
            frame, stamp = camera_node.get_frame_and_stamp()

            # Skip if no new frame has arrived since the last iteration
            # (keeps us from saving duplicate images faster than AMBF publishes).
            if frame is None or stamp == last_stamp:
                time.sleep(0.005)
                continue
            last_stamp = stamp

            augmented, labels, status = augmentor.augment(frame)
            if augmented is None:
                print(f"  [skip] {status}")
                continue

            img_path = os.path.join(OUTPUT_DIR, f"frame_{saved:05d}.png")
            cv2.imwrite(img_path, augmented)

            if SAVE_LABELS and labels is not None:
                label_path = os.path.join(OUTPUT_DIR, f"frame_{saved:05d}.json")
                with open(label_path, "w") as f:
                    json.dump(labels, f, indent=2)

            saved += 1
            if saved % 25 == 0 or saved == NUM_IMAGES_TO_GENERATE:
                print(f"  [{saved}/{NUM_IMAGES_TO_GENERATE}] "
                      f"{os.path.basename(img_path)}  ({status})")

            if SHOW_WINDOW:
                try:
                    cv2.imshow("Landmark Jitter Augmentor", augmented)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

    except KeyboardInterrupt:
        print("\n[LOOP] Interrupted by user — stopping early.")

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"  Done. Saved {saved} augmented frames to: {OUTPUT_DIR}")
    print("=" * 70)
    rclpy.shutdown()


if __name__ == "__main__":
    main()