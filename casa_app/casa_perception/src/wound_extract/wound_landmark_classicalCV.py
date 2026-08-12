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
                 wound image.
    4. SEGMENT   Find the wound itself via a local texture/roughness signal
                 (see "WHY TEXTURE, NOT COLOR" below), independent of zoom
                 level or wound orientation.
    5. PLACE     Extracts the wound's centerline and drops landmarks in
                 FLANKING PAIRS on either side of it — matching how the
                 original markers are actually arranged — spaced evenly
                 along its length.
    6. RE-DRAW   Pastes each new landmark using a REAL patch borrowed from
                 whichever original marker is closest to it, so size/
                 shading stays consistent with the surrounding scene.

WHY TEXTURE, NOT COLOR, FOR THE WOUND
---------------------------------------
The wound is the SAME hue/saturation as the surrounding tissue (measured:
both around H=150-160, S=110-135) — it's a rough, stitched groove, not a
differently-colored region. So step 4 looks for local pixel variance
("roughness") instead of a color difference.

ABOUT THE DEBUG IMAGE (02_wound_mask_debug.png)
-------------------------------------------------
The green overlay is a DIAGNOSTIC, not part of the final output — it's
every pixel the algorithm currently classifies as "wound texture," so you
can visually sanity-check the segmentation. It normally looks WIDER than
the thin visible seam, because it's genuinely capturing the whole rough/
feathered stitch pattern on both sides of the centerline, not just a
single line down the middle — that's expected. If it ever looks
completely wrong (covering unrelated areas, or empty), that's the signal
something needs re-tuning, not this image itself being an error.

GENERALIZING ACROSS CAMERA POSITIONS
--------------------------------------
This version is built to be scale- and rotation-invariant, not hard-coded
to one framing:
  - Every geometric constant (erosion size, texture window, closing
    kernel, distance gates) is computed as a FRACTION of the tissue
    block's own detected size in the current frame, not a fixed pixel
    count — so it adapts automatically whether the camera is zoomed in
    or out.
  - The wound is picked out with PCA-based elongation (major/minor axis
    ratio), not a "must be taller than wide" bounding-box check — so it
    still works if the camera angle makes the wound appear more
    horizontal or diagonal, not just vertical.
  - The centerline is extracted by projecting onto the wound's own
    principal axis (via PCA), not by scanning image rows — so it holds
    up regardless of the wound's orientation in the frame.
  - The flanking offset (how far each landmark sits from the centerline)
    is MEASURED from the real markers already in the current frame, not
    a fixed pixel value — so it automatically matches whatever scale
    that frame happens to be at.

This gets you invariance to zoom and in-plane rotation for this same
phantom under similar lighting — the two real captures at very different
zoom levels behaved consistently once this was in place. It does NOT
by itself protect against a fundamentally different lighting rig, a
different-colored phantom, tools occluding the wound, or extreme
near-edge-on viewing angles — classical, hand-tuned CV has a real ceiling
there. For that level of robustness you'd want a learned model (e.g. the
U-Net you asked about earlier), trained on a variety of captured views.
This script's texture-based wound masks are a reasonable way to bootstrap
that training data later — run it across many captured frames/angles and
use the resulting masks as an initial labeled set to hand-correct.

OUTPUT
------
    output/00_original.png            the raw captured frame
    output/01_markers_removed.png     clean tissue + wound, no markers
    output/02_wound_mask_debug.png    diagnostic: wound mask overlay
    output/03_final_output.png        the deliverable: clean wound with
                                       new landmarks placed (flanking)
    output/03_final_output.json       the new landmark (x, y) coordinates

HOW TO RUN
----------
    source ~/ros2_ws/install/setup.bash
    python3 wound_landmark_classicalCV.py
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
NUM_LANDMARKS = 8  # total landmarks to place (must be even -> NUM_LANDMARKS/2 pairs)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SAVE_DEBUG_IMAGES = True
SAVE_LABELS = True

# --- tissue mask (the phantom block itself) ---------------------------------
TISSUE_HSV_LOWER = np.array([140, 40, 0])
TISSUE_HSV_UPPER = np.array([179, 255, 255])

# --- marker detection (the existing red/orange squares) ---------------------
MARKER_HSV_RANGES = [
    {"lower": np.array([170, 160, 0]), "upper": np.array([179, 255, 255])},
    {"lower": np.array([0, 160, 0]),   "upper": np.array([8, 255, 255])},
]
MIN_MARKER_AREA = 5
PATCH_PADDING = 4

# --- inpainting (marker removal) --------------------------------------------
INPAINT_RADIUS = 4
INPAINT_METHOD = cv2.INPAINT_TELEA
INPAINT_MASK_DILATE_PX = 5
INPAINT_MASK_DILATE_ITERS = 2

# --- wound segmentation (all scaled off the tissue block's own size) --------
TISSUE_ERODE_FRACTION = 0.02     # keeps block edges out of the texture analysis
TEXTURE_WINDOW_FRACTION = 0.012  # local-variance window, as a fraction of tissue diagonal
TEXTURE_PERCENTILE = 90          # pixels above this local-std percentile = "textured"
WOUND_CLOSE_FRACTION = 0.02      # morphological closing kernel size
WOUND_MIN_AREA_FRACTION = 0.01   # candidate blobs smaller than this (of tissue diagonal^2) are noise
WOUND_MIN_ELONGATION = 2.5       # PCA major/minor axis ratio -- filters out non-wound-shaped blobs
WOUND_MAX_MARKER_DIST_FRACTION = 0.35  # candidate must be within this fraction of the tissue
                                          # diagonal from the markers' centroid

# --- re-draw ------------------------------------------------------------------
FEATHER_BLUR_KSIZE = 3


# ==============================================================================
# ROS2 CAMERA SUBSCRIBER
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
    blob. This also gives us a built-in ruler: the block's own size in this
    frame, which every downstream threshold is scaled against."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TISSUE_HSV_LOWER, TISSUE_HSV_UPPER)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.uint8(labels == largest) * 255


def tissue_scale(tissue_mask: np.ndarray) -> float:
    """The tissue block's bounding-box diagonal, in pixels. Used as the
    reference length for every other scale-adaptive threshold, so the
    pipeline behaves the same whether the camera is zoomed in or out."""
    x, y, w, h = cv2.boundingRect(tissue_mask)
    return float(np.hypot(w, h))


def detect_markers(frame: np.ndarray, tissue_mask: np.ndarray):
    """Finds the existing red/orange landmark squares inside the tissue
    region. Returns a list of dicts (center, bbox, local shape mask, and a
    cropped pixel patch) plus the raw marker mask."""
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
    """Erases the detected markers with cv2.inpaint()."""
    dilate_kernel = np.ones((INPAINT_MASK_DILATE_PX, INPAINT_MASK_DILATE_PX), np.uint8)
    dilated = cv2.dilate(marker_mask, dilate_kernel, iterations=INPAINT_MASK_DILATE_ITERS)
    return cv2.inpaint(frame, dilated, INPAINT_RADIUS, INPAINT_METHOD)


# ==============================================================================
# STEP 4 — WOUND SEGMENTATION (texture-based, scale- and rotation-invariant)
# ==============================================================================

def segment_wound(clean_frame: np.ndarray, tissue_mask: np.ndarray,
                   diag: float, markers_centroid=None) -> np.ndarray:
    """
    Finds the wound band on the (marker-free) tissue using local texture
    ("roughness"), since the wound is the same color as the tissue around
    it. Every size threshold below is a FRACTION of `diag` (the tissue
    block's own bounding-box diagonal in this frame) rather than a fixed
    pixel count, so this adapts automatically to zoom level.

    Candidates are then filtered by PCA-based elongation (works for a
    wound at any angle, not just vertical) and, if available, how close
    they are to the removed markers' centroid (the markers always flank
    the real wound) — among survivors, the LARGEST wins, since a tiny
    stray textured speck can otherwise be closer-but-wrong.
    """
    erode_px = max(3, int(TISSUE_ERODE_FRACTION * diag))
    tissue_core = cv2.erode(tissue_mask, np.ones((erode_px, erode_px), np.uint8))

    win = max(5, int(TEXTURE_WINDOW_FRACTION * diag))
    if win % 2 == 0:
        win += 1
    gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (win, win))
    mean_sq = cv2.blur(gray * gray, (win, win))
    local_std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None)) * (tissue_core > 0)

    valid = local_std[tissue_core > 0]
    if valid.size == 0:
        return np.zeros(tissue_mask.shape, dtype=np.uint8)
    thresh_val = np.percentile(valid, TEXTURE_PERCENTILE)
    wound_mask = np.uint8(local_std > thresh_val) * 255

    close_px = max(3, int(WOUND_CLOSE_FRACTION * diag))
    if close_px % 2 == 0:
        close_px += 1
    wound_mask = cv2.morphologyEx(
        wound_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    )
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, k3)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(wound_mask, 8)
    min_area = max(30.0, (WOUND_MIN_AREA_FRACTION * diag) ** 2)
    max_dist = WOUND_MAX_MARKER_DIST_FRACTION * diag

    best_label, best_area = None, -1.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) < 5:
            continue
        mean_pt, eigvecs, eigvals = cv2.PCACompute2(pts, mean=np.array([]))
        eigvals = eigvals.flatten()
        elongation = np.sqrt(max(eigvals[0], 1e-6) / max(eigvals[1], 1e-6))
        if elongation < WOUND_MIN_ELONGATION:
            continue

        if markers_centroid is not None:
            cx, cy = mean_pt[0]
            dist = float(np.hypot(cx - markers_centroid[0], cy - markers_centroid[1]))
            if dist > max_dist:
                continue

        if area > best_area:
            best_area, best_label = area, i

    if best_label is None:
        return np.zeros(tissue_mask.shape, dtype=np.uint8)
    return np.uint8(labels == best_label) * 255


def extract_centerline(wound_mask: np.ndarray, n_bins: int = 150) -> np.ndarray:
    """
    Returns an ordered (N, 2) array of (x, y) points down the middle of the
    wound mask. Uses PCA to find the blob's own principal axis and
    projects all its pixels onto that axis, so this works regardless of
    whether the wound appears vertical, horizontal, or diagonal in frame
    (unlike scanning image rows, which only works for near-vertical wounds).
    """
    ys, xs = np.nonzero(wound_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) < 2:
        return pts

    mean_pt, eigvecs = cv2.PCACompute(pts, mean=np.array([]))
    principal = eigvecs[0]
    t = (pts - mean_pt[0]) @ principal
    order = np.argsort(t)
    t_sorted, pts_sorted = t[order], pts[order]

    bins = np.linspace(t_sorted.min(), t_sorted.max(), n_bins + 1)
    centerline = []
    for i in range(n_bins):
        sel = (t_sorted >= bins[i]) & (t_sorted < bins[i + 1])
        if np.any(sel):
            centerline.append(pts_sorted[sel].mean(axis=0))
    return np.array(centerline)


# ==============================================================================
# STEP 5 — LANDMARK PLACEMENT: flanking pairs, spaced along the centerline
# ==============================================================================

def compute_flanking_offset(markers: list, centerline: np.ndarray) -> float:
    """
    Measures how far the REAL original markers sit from the wound
    centerline (their median perpendicular-ish distance to the nearest
    centerline point) and reuses that as the offset for new landmarks.
    Because it's measured from the current frame's own markers, this
    scales automatically with zoom level -- no fixed pixel constant needed.
    """
    if not markers or len(centerline) == 0:
        return 20.0  # fallback if no markers were found to measure from
    offsets = []
    for m in markers:
        mx, my = m["center"]
        d = np.hypot(centerline[:, 0] - mx, centerline[:, 1] - my)
        offsets.append(float(d.min()))
    return float(np.median(offsets))


def sample_flanking_pairs(centerline: np.ndarray, num_pairs: int, offset: float,
                           markers: list, canvas_shape: tuple) -> list:
    """
    Places `num_pairs` rows at EXACTLY equal Euclidean (straight-line)
    distance from each other, guaranteed by linear interpolation between
    the wound's two endpoints.

    The endpoints are pulled inward first by a safety margin (based on
    the largest marker patch we might paste there) so that no landmark
    -- especially the first/last row, which sit closest to the frame
    edge -- ever gets its patch cropped by the image boundary. A cropped
    patch's visible portion has a different centroid than its true
    center, which throws off the actual on-screen spacing even when the
    underlying target coordinates are perfectly even.
    """
    if len(centerline) < 2:
        return []

    order = np.argsort(centerline[:, 1])
    ordered = centerline[order]
    p_start = ordered[0]
    p_end = ordered[-1]

    direction = p_end - p_start
    total_len = np.hypot(*direction)
    direction = direction / total_len if total_len > 1e-6 else np.array([0.0, 1.0])
    perpendicular = np.array([-direction[1], direction[0]])

    if markers:
        max_half_diag = max(
            np.hypot(m["bbox"][2] - m["bbox"][0], m["bbox"][3] - m["bbox"][1]) / 2.0
            for m in markers
        )
    else:
        max_half_diag = 0.0
    margin = min(max_half_diag, 0.4 * total_len)  # cap so short wounds don't collapse

    p_start = p_start + direction * margin
    p_end = p_end - direction * margin

    pairs = []
    for i in range(num_pairs):
        t = i / (num_pairs - 1) if num_pairs > 1 else 0.0
        p = p_start + t * (p_end - p_start)
        pairs.append((tuple(p - perpendicular * offset), tuple(p + perpendicular * offset)))
    return pairs


# ==============================================================================
# STEP 6 — RE-DRAW: paste new landmarks using real marker pixels
# ==============================================================================

def _nearest_marker(markers: list, point: tuple) -> dict:
    """Finds the original marker closest (in image distance) to `point`,
    so its patch can be reused for a new landmark placed there."""
    px, py = point
    return min(markers, key=lambda m: np.hypot(m["center"][0] - px, m["center"][1] - py))


def draw_landmark(canvas: np.ndarray, template_marker: dict, new_center: tuple) -> None:
    """
    Pastes a template marker's real pixels centered EXACTLY at new_center.

    If the patch would extend past the image edge (this happens for
    landmarks near the top/bottom of frame, especially larger patches
    borrowed from markers close to the camera), only the out-of-bounds
    sliver is cropped -- the patch is NOT shifted inward to force it
    on-canvas. Shifting would silently move the marker's visual center
    away from the intended coordinate (this was a real bug: a landmark
    intended at y=477 was rendered at y~448 because its patch didn't fit
    before the bottom edge, throwing off the actual on-screen spacing
    even though the computed coordinates were correct).
    """
    x0, y0, x1, y1 = template_marker["bbox"]
    w, h = x1 - x0, y1 - y0
    H, W = canvas.shape[:2]

    tgt_x0 = int(round(new_center[0] - w / 2))
    tgt_y0 = int(round(new_center[1] - h / 2))
    tgt_x1, tgt_y1 = tgt_x0 + w, tgt_y0 + h

    clip_x0, clip_y0 = max(tgt_x0, 0), max(tgt_y0, 0)
    clip_x1, clip_y1 = min(tgt_x1, W), min(tgt_y1, H)
    if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
        return  # marker's center itself is off-canvas -- nothing valid to draw

    src_x0, src_y0 = clip_x0 - tgt_x0, clip_y0 - tgt_y0
    src_x1, src_y1 = src_x0 + (clip_x1 - clip_x0), src_y0 + (clip_y1 - clip_y0)

    alpha_full = cv2.GaussianBlur(
        template_marker["local_mask"].astype(np.float32) / 255.0,
        (FEATHER_BLUR_KSIZE, FEATHER_BLUR_KSIZE), 0,
    )
    alpha = alpha_full[src_y0:src_y1, src_x0:src_x1][..., None]
    patch = template_marker["patch"][src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    region = canvas[clip_y0:clip_y1, clip_x0:clip_x1].astype(np.float32)
    canvas[clip_y0:clip_y1, clip_x0:clip_x1] = (region * (1 - alpha) + patch * alpha).astype(np.uint8)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("  CASA — Wound Segmentation & Landmark Placement Pipeline")
    print("=" * 70)

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

    print("\n[STEP 2] Detecting existing landmark markers...")
    tissue_mask = segment_tissue(frame)
    diag = tissue_scale(tissue_mask)
    markers, marker_mask = detect_markers(frame, tissue_mask)
    print(f"[STEP 2] Found {len(markers)} marker(s). Tissue scale (diag): {diag:.0f}px")

    print("[STEP 3] Removing markers with cv2.inpaint()...")
    clean_frame = remove_markers(frame, marker_mask)
    if SAVE_DEBUG_IMAGES:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "01_markers_removed.png"), clean_frame)

    print("[STEP 4] Segmenting the wound (texture-based, scale-adaptive)...")
    markers_centroid = (
        np.mean([m["center"] for m in markers], axis=0) if markers else None
    )
    wound_mask = segment_wound(clean_frame, tissue_mask, diag, markers_centroid)
    wound_area = int((wound_mask > 0).sum())
    if wound_area == 0:
        print("[ERROR] Could not find the wound — check the TISSUE_*/WOUND_* "
              "fractions against a fresh sample frame.")
        rclpy.shutdown()
        sys.exit(1)
    print(f"[STEP 4] Wound segmented ({wound_area} px).")

    if SAVE_DEBUG_IMAGES:
        overlay = clean_frame.copy()
        overlay[wound_mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(clean_frame, 0.6, overlay, 0.4, 0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "02_wound_mask_debug.png"), blended)

    print(f"[STEP 5] Placing {NUM_LANDMARKS} landmarks flanking the wound...")
    centerline = extract_centerline(wound_mask)
    offset = compute_flanking_offset(markers, centerline)
    num_pairs = max(1, NUM_LANDMARKS // 2)
    pairs = sample_flanking_pairs(centerline, num_pairs, offset, markers, frame.shape)
    print(f"[STEP 5] Flanking offset measured at {offset:.1f}px from this frame's own markers.")

    # Explicit proof, printed right here: straight-line distance between
    # consecutive row centers. These MUST match -- that's not a claim,
    # it's guaranteed by the linear-interpolation construction above.
    row_centers = [ ((l[0]+r[0])/2, (l[1]+r[1])/2) for l, r in pairs ]
    row_dists = [
        float(np.hypot(row_centers[i+1][0]-row_centers[i][0], row_centers[i+1][1]-row_centers[i][1]))
        for i in range(len(row_centers) - 1)
    ]
    print(f"[STEP 5] Row centers: {[(round(x,1), round(y,1)) for x,y in row_centers]}")
    print(f"[STEP 5] Distance between consecutive rows (px): {[round(d, 2) for d in row_dists]}")

    print("[STEP 6] Drawing new landmarks with real marker pixels...")
    final_image = clean_frame.copy()
    labels_out = []
    for left, right in pairs:
        for point, side in ((left, "left"), (right, "right")):
            if markers:
                template = _nearest_marker(markers, point)
                draw_landmark(final_image, template, point)
            labels_out.append({"x": round(float(point[0]), 1),
                                "y": round(float(point[1]), 1),
                                "side": side})

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

    # Clean shutdown: destroy the node and let rclpy.shutdown() cause the
    # background spin() call to return on its own, then join the thread.
    # Skipping this (just calling rclpy.shutdown() and letting the daemon
    # thread get killed at process exit) is what caused the
    # "terminate called without an active exception / Aborted (core dumped)"
    # crash -- the DDS middleware doesn't like being killed mid-call.
    camera_node.destroy_node()
    rclpy.shutdown()
    spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()