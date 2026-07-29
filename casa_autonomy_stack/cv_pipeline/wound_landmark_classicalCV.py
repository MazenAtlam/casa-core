#!/usr/bin/env python3
"""
================================================================================
  wound_landmark_classicalCV.py
  CASA Surgical Robotics Challenge — Wound Segmentation & Landmark Placement
================================================================================

PURPOSE
-------
Takes ONE frame from the live AMBF camera feed and turns it into a clean,
auto-labeled "start state" image for the downstream imitation-learning
model. Pipeline:

    1. CAPTURE   Grab one frame from /ambf/env/stereo/left/ImageData when
                 the simulator starts (ROS2 is started exactly once).
    2. DETECT    Locate the 8 existing red/orange landmark markers.
    3. REMOVE    Erase them with cv2.inpaint(), leaving a clean tissue +
                 wound image (no training data / manual masks needed for
                 this step — pure color thresholding, same approach as
                 the earlier jitter-augmentor script).
    4. SEGMENT   Find the wound itself. The wound is the SAME color as the
                 surrounding tissue (same hue/saturation) — it's a
                 textured groove, not a differently-colored region — so
                 plain color thresholding won't isolate it. Instead this
                 uses a classical, no-training texture/edge-energy signal
                 (local pixel variance) to find the "rough" band running
                 down the tissue, then keeps only the tall/narrow blob
                 nearest to where the removed markers used to be.
    5. PLACE     Extracts the wound's centerline and drops NUM_LANDMARKS
                 new markers evenly spaced along it.
    6. RE-DRAW   Pastes each new landmark using a REAL patch borrowed from
                 one of the original (now-removed) markers — matched by
                 vertical position, so size/perspective stays consistent
                 down the length of the wound — for a photorealistic
                 result with no synthetic-looking shapes.

WHY NOT U-NET HERE?
--------------------
U-Net (or any supervised segmentation model) needs a training set of
(image, ground-truth mask) pairs, which isn't available yet. This script
uses classical CV instead — no training required, works today. If you
later want to move to U-Net (e.g. to generalize past this exact
simulator/phantom), the wound masks this script produces internally
(see SAVE_DEBUG_IMAGES) can double as a starting point for auto-generated
training labels.

NOTE ON TUNING
--------------
The HSV / texture thresholds below were tuned against a real captured
frame from this phantom. If you change the phantom's material color,
lighting, or camera framing significantly, re-check these values against
a fresh sample frame the same way (sample pixel HSV values at tissue vs.
marker vs. wound locations) before trusting the output.

OUTPUT
------
    output/00_original.png            the raw captured frame
    output/01_markers_removed.png     clean tissue + wound, no markers
    output/02_wound_mask_debug.png    visual check: wound mask overlay
    output/03_final_output.png        the deliverable: clean wound with
                                       new landmarks placed
    output/03_final_output.json       the new landmark (x, y) coordinates
                                       (the label the imitation-learning
                                       model will train against)

HOW TO RUN
----------
    source ~/ros2_ws/install/setup.bash
    python3 wound_landmark_pipeline.py
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
# CONFIGURATION
# ==============================================================================

CAMERA_TOPIC = "/ambf/env/stereo/left/ImageData"
NUM_LANDMARKS = 8  # how many new landmarks to place along the wound

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SAVE_DEBUG_IMAGES = True   # saves the intermediate steps too, not just the final image
SAVE_LABELS = True         # saves a .json with the new landmark coordinates

# --- tissue mask (the phantom block itself) ---------------------------------
# Sampled from a real frame: the magenta tissue sits around H=150-160, S=110-135,
# with V varying by lighting. The grey floor/background has near-zero saturation,
# so a saturation floor is enough to separate tissue from everything else.
TISSUE_HSV_LOWER = np.array([140, 40, 0])
TISSUE_HSV_UPPER = np.array([179, 255, 255])
TISSUE_ERODE_PX = 15  # shrinks the tissue mask inward so block EDGES (which also
                       # have high local texture/contrast) don't get mistaken
                       # for the wound in the texture-based segmentation step

# --- marker detection (the existing red/orange squares) ---------------------
# Markers have a distinctly higher saturation / different hue than the
# surrounding magenta tissue (a real, measured spike: S~200+ vs tissue S~110-135).
MARKER_HSV_RANGES = [
    {"lower": np.array([170, 160, 0]), "upper": np.array([179, 255, 255])},
    {"lower": np.array([0, 160, 0]),   "upper": np.array([8, 255, 255])},
]
MIN_MARKER_AREA = 5
PATCH_PADDING = 4  # px of context captured around each marker for re-pasting later

# --- inpainting (marker removal) --------------------------------------------
INPAINT_RADIUS = 4
INPAINT_METHOD = cv2.INPAINT_TELEA
INPAINT_MASK_DILATE_PX = 5
INPAINT_MASK_DILATE_ITERS = 2

# --- wound (texture-based) segmentation --------------------------------------
TEXTURE_WINDOW = 9          # local-variance window size, px
TEXTURE_PERCENTILE = 90     # pixels above this local-std percentile = "textured"
WOUND_CLOSE_KERNEL = (5, 11)  # (w, h) — taller than wide, matches a vertical wound
WOUND_MIN_ASPECT = 2.0      # candidate blobs must be at least this much taller than wide

# --- re-draw ------------------------------------------------------------------
FEATHER_BLUR_KSIZE = 3  # softens the pasted patch edge so it blends in


# ==============================================================================
# ROS2 CAMERA SUBSCRIBER — started once, per the project convention
# ==============================================================================

class AmbfCameraSubscriber(Node):
    """Subscribes once to the AMBF left camera topic and holds onto the most
    recent frame for the main script to grab when it's ready."""

    def __init__(self):
        super().__init__("wound_landmark_pipeline")
        self.bridge = CvBridge()
        self.latest_frame = None
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
            with self._lock:
                self.latest_frame = cv_image.copy()
        except CvBridgeError as e:
            self.get_logger().error(f"[CvBridge Error] {e}")

    def get_frame(self):
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()


# ==============================================================================
# STEP 2/3 — TISSUE MASK + MARKER DETECTION + REMOVAL
# ==============================================================================

def segment_tissue(frame: np.ndarray) -> np.ndarray:
    """Isolates the phantom block from the grey background as one connected
    blob — this defines the region of interest for everything downstream."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TISSUE_HSV_LOWER, TISSUE_HSV_UPPER)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask  # nothing found — return as-is, caller should handle
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.uint8(labels == largest) * 255


def detect_markers(frame: np.ndarray, tissue_mask: np.ndarray) -> list:
    """Finds the existing red/orange landmark squares inside the tissue
    region. Returns a list of dicts (one per marker) with center, bbox,
    a local shape mask, and a cropped pixel patch — everything needed to
    both erase them and later re-paste new markers using their real look."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    for r in MARKER_HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r["lower"], r["upper"]))
    mask = cv2.bitwise_and(mask, tissue_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_MARKER_AREA]

    markers = []
    for c in contours:
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(c)
        x0, y0 = max(x - PATCH_PADDING, 0), max(y - PATCH_PADDING, 0)
        x1, y1 = min(x + w + PATCH_PADDING, W), min(y + h + PATCH_PADDING, H)

        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(local_mask, [c], -1, 255, thickness=-1, offset=(-x0, -y0))

        markers.append({
            "center": (cx, cy),
            "bbox": (x0, y0, x1, y1),
            "local_mask": local_mask,
            "patch": frame[y0:y1, x0:x1].copy(),
        })

    return markers, mask


def remove_markers(frame: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
    """Erases the detected markers with cv2.inpaint(), leaving a clean
    tissue + wound image underneath."""
    dilate_kernel = np.ones((INPAINT_MASK_DILATE_PX, INPAINT_MASK_DILATE_PX), np.uint8)
    dilated = cv2.dilate(marker_mask, dilate_kernel, iterations=INPAINT_MASK_DILATE_ITERS)
    return cv2.inpaint(frame, dilated, INPAINT_RADIUS, INPAINT_METHOD)


# ==============================================================================
# STEP 4 — WOUND SEGMENTATION (classical CV, no training)
# ==============================================================================

def segment_wound(clean_frame: np.ndarray, tissue_mask: np.ndarray,
                   marker_center_x_hint: float = None) -> np.ndarray:
    """
    Finds the wound band on the (marker-free) tissue.

    The wound is textured (a rough groove/stitch pattern) but the SAME
    color as the surrounding smooth tissue, so this looks for local pixel
    variance instead of a color difference. Among the resulting candidate
    blobs, it keeps the tall/narrow ones (aspect ratio filter) and — if a
    hint is available from where the original markers were — picks the
    one closest to that column, since the markers always flank the wound.
    """
    tissue_core = cv2.erode(tissue_mask, np.ones((TISSUE_ERODE_PX, TISSUE_ERODE_PX), np.uint8))

    gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (TEXTURE_WINDOW, TEXTURE_WINDOW))
    mean_sq = cv2.blur(gray * gray, (TEXTURE_WINDOW, TEXTURE_WINDOW))
    local_std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    local_std = local_std * (tissue_core > 0)

    valid = local_std[tissue_core > 0]
    if valid.size == 0:
        return np.zeros(tissue_mask.shape, dtype=np.uint8)

    thresh_val = np.percentile(valid, TEXTURE_PERCENTILE)
    wound_mask = np.uint8(local_std > thresh_val) * 255
    wound_mask = cv2.morphologyEx(
        wound_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, WOUND_CLOSE_KERNEL)
    )
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, k3)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(wound_mask, 8)
    best_label, best_score = None, None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < WOUND_MIN_ASPECT * w:
            continue  # not tall/narrow enough to be the wound
        cx = x + w / 2.0
        dist = abs(cx - marker_center_x_hint) if marker_center_x_hint is not None else 0.0
        score = (dist, -area)  # closest to the marker column, then largest
        if best_score is None or score < best_score:
            best_score, best_label = score, i

    if best_label is None:
        return np.zeros(tissue_mask.shape, dtype=np.uint8)

    return np.uint8(labels == best_label) * 255


def extract_centerline(wound_mask: np.ndarray) -> np.ndarray:
    """Returns an ordered (N, 2) array of (x, y) points down the middle of
    the wound mask — one point per row that contains wound pixels."""
    rows_with_wound = np.where(wound_mask.any(axis=1))[0]
    points = []
    for y in rows_with_wound:
        xs = np.where(wound_mask[y] > 0)[0]
        points.append((xs.mean(), float(y)))
    return np.array(points)


def sample_evenly_along_centerline(centerline: np.ndarray, n: int) -> list:
    """Picks n points evenly spaced by ARC LENGTH along the centerline
    (not just evenly spaced by row index), so spacing looks natural even
    if the wound isn't perfectly straight."""
    if len(centerline) < 2:
        return [tuple(p) for p in centerline]

    diffs = np.diff(centerline, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    sample_targets = np.linspace(0, total_len, n)
    points = []
    for target in sample_targets:
        idx = min(np.searchsorted(cum_len, target), len(centerline) - 1)
        points.append(tuple(centerline[idx]))
    return points


# ==============================================================================
# STEP 6 — RE-DRAW: paste new landmarks using real marker pixels
# ==============================================================================

def _nearest_marker_by_y(markers: list, target_y: float) -> dict:
    """Finds the original marker whose vertical position is closest to
    target_y, so its patch (size/shading matching that part of the wound
    due to perspective) can be reused for a new landmark placed there."""
    return min(markers, key=lambda m: abs(m["center"][1] - target_y))


def draw_landmark(canvas: np.ndarray, template_marker: dict, new_center: tuple) -> None:
    """Pastes a template marker's real pixels at a new (x, y) location,
    using its exact shape as a feathered alpha mask so it blends in
    cleanly instead of leaving a hard edge."""
    x0, y0, x1, y1 = template_marker["bbox"]
    w, h = x1 - x0, y1 - y0
    old_cx, old_cy = template_marker["center"]

    new_x0 = int(round(new_center[0] - w / 2))
    new_y0 = int(round(new_center[1] - h / 2))
    new_x1, new_y1 = new_x0 + w, new_y0 + h

    H, W = canvas.shape[:2]
    new_x0 = int(np.clip(new_x0, 0, W - w))
    new_y0 = int(np.clip(new_y0, 0, H - h))
    new_x1, new_y1 = new_x0 + w, new_y0 + h

    alpha = cv2.GaussianBlur(
        template_marker["local_mask"].astype(np.float32) / 255.0,
        (FEATHER_BLUR_KSIZE, FEATHER_BLUR_KSIZE), 0,
    )[..., None]

    region = canvas[new_y0:new_y1, new_x0:new_x1].astype(np.float32)
    patch = template_marker["patch"].astype(np.float32)
    canvas[new_y0:new_y1, new_x0:new_x1] = (region * (1 - alpha) + patch * alpha).astype(np.uint8)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("  CASA — Wound Segmentation & Landmark Placement Pipeline")
    print("=" * 70)

    # --- Start ROS2 exactly once ---
    print("\n[STEP 1] Initializing ROS2...")
    rclpy.init()
    camera_node = AmbfCameraSubscriber()
    spin_thread = threading.Thread(target=rclpy.spin, args=(camera_node,), daemon=True)
    spin_thread.start()
    print(f"[STEP 1] Subscribed to {CAMERA_TOPIC}. Waiting for a frame...")

    timeout_seconds = 30
    frame = None
    start_time = time.time()
    while frame is None:
        frame = camera_node.get_frame()
        if frame is None:
            if time.time() - start_time > timeout_seconds:
                print(f"[ERROR] No frame received after {timeout_seconds}s. "
                      f"Is the AMBF simulator running?")
                rclpy.shutdown()
                sys.exit(1)
            time.sleep(0.5)
    print(f"[STEP 1] Frame captured. Shape: {frame.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_DEBUG_IMAGES:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "00_original.png"), frame)

    # --- Steps 2/3: detect + remove the existing markers ---
    print("\n[STEP 2] Detecting existing landmark markers...")
    tissue_mask = segment_tissue(frame)
    markers, marker_mask = detect_markers(frame, tissue_mask)
    print(f"[STEP 2] Found {len(markers)} marker(s).")

    print("[STEP 3] Removing markers with cv2.inpaint()...")
    clean_frame = remove_markers(frame, marker_mask)
    if SAVE_DEBUG_IMAGES:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "01_markers_removed.png"), clean_frame)

    # --- Step 4: segment the wound itself ---
    print("[STEP 4] Segmenting the wound (texture-based, no training)...")
    marker_center_x_hint = (
        float(np.mean([m["center"][0] for m in markers])) if markers else None
    )
    wound_mask = segment_wound(clean_frame, tissue_mask, marker_center_x_hint)
    wound_area = int((wound_mask > 0).sum())
    if wound_area == 0:
        print("[ERROR] Could not find the wound — check TEXTURE_* / TISSUE_* "
              "thresholds against a fresh sample frame.")
        rclpy.shutdown()
        sys.exit(1)
    print(f"[STEP 4] Wound segmented ({wound_area} px).")

    if SAVE_DEBUG_IMAGES:
        overlay = clean_frame.copy()
        overlay[wound_mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(clean_frame, 0.6, overlay, 0.4, 0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "02_wound_mask_debug.png"), blended)

    # --- Step 5: place new landmarks along the wound's centerline ---
    print(f"[STEP 5] Placing {NUM_LANDMARKS} landmarks along the wound centerline...")
    centerline = extract_centerline(wound_mask)
    new_positions = sample_evenly_along_centerline(centerline, NUM_LANDMARKS)

    # --- Step 6: re-draw using real marker pixels ---
    print("[STEP 6] Drawing new landmarks with real marker pixels...")
    final_image = clean_frame.copy()
    labels_out = []
    for (x, y) in new_positions:
        if markers:
            template = _nearest_marker_by_y(markers, y)
            draw_landmark(final_image, template, (x, y))
        labels_out.append({"x": round(float(x), 1), "y": round(float(y), 1)})

    final_path = os.path.join(OUTPUT_DIR, "03_final_output.png")
    cv2.imwrite(final_path, final_image)

    if SAVE_LABELS:
        label_path = os.path.join(OUTPUT_DIR, "03_final_output.json")
        with open(label_path, "w") as f:
            json.dump({"landmarks": labels_out}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Done. Final image saved to: {final_path}")
    if SAVE_LABELS:
        print(f"  Labels saved to: {label_path}")
    if SAVE_DEBUG_IMAGES:
        print(f"  Intermediate steps also saved in: {OUTPUT_DIR}")
    print("=" * 70)

    rclpy.shutdown()


if __name__ == "__main__":
    main()