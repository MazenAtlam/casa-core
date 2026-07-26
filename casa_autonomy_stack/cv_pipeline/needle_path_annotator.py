#!/usr/bin/env python3
"""
================================================================================
  needle_path_annotator.py
  CASA Surgical Robotics Challenge — Computer Vision Pipeline
================================================================================

PURPOSE
-------
This script is the main CV pipeline for the CASA project. It:
    1. Subscribes to the AMBF simulator's mono camera (left stereo camera)
    2. Captures a single frame from the simulation
    3. Runs tissue segmentation to identify the surgical phantom
    4. Predicts needle path waypoints (red dots) on the segmented region
    5. Draws those red dots on the original image and saves it to disk

These output images will later be used as training labels for an imitation
learning model operated by another team member.

PIPELINE OVERVIEW
-----------------
    AMBF Simulator
         │
         │  ROS2 topic: /ambf/env/stereo/left/ImageData
         │  Message type: sensor_msgs/Image (640x480, RGB)
         ▼
    [STEP 1] ROS2 Subscriber  →  captures raw image frame
         │
         ▼
    [STEP 2] CvBridge conversion  →  converts ROS Image msg → OpenCV/NumPy array
         │
         ▼
    [STEP 3] HSV Color Segmentation  →  identifies surgical tissue mask
         │                              (placeholder — swap to U-Net when trained)
         ▼
    [STEP 4] Keypoint Prediction  →  predicts needle path waypoints from mask
         │
         ▼
    [STEP 5] Annotation & Save  →  draws red dots, saves final image to disk

OUTPUT
------
    Annotated images are saved to:
        ~/ros2_ws/src/casa-core/casa_autonomy_stack/cv_pipeline/output/

    Filename format:
        needle_path_YYYYMMDD_HHMMSS.png

    The raw (unannotated) camera frame is also saved alongside it:
        raw_frame_YYYYMMDD_HHMMSS.png

HOW TO RUN
----------
    Make sure the AMBF simulator is already running:
        cd ~/ros2_ws/src/casa-core/casa_app/surgical_robotics_challenge
        ./run_env_SIMPLE_LND_420006.sh

    In a NEW terminal, source and run:
        source ~/ros2_ws/install/setup.bash
        python3 ~/ros2_ws/src/casa-core/casa_autonomy_stack/cv_pipeline/needle_path_annotator.py

DEPENDENCIES
------------
    pip install opencv-python-headless numpy
    sudo apt install ros-humble-cv-bridge
    (No PyTorch needed — this version uses OpenCV-based segmentation)

CAMERA TOPIC INFO (from world_stereo.yaml)
------------------------------------------
    Topic  : /ambf/env/stereo/left/ImageData
    Encoding: bgr8 (converted by CvBridge)
    Resolution: 640 x 480 pixels
    Field of View: 0.9599 radians (~55 degrees)
    Published every 5 simulation ticks

    The topic name is built from the AMBF world config:
        namespace:  /ambf/env/       (from world_stereo.yaml line 5)
        camera ns:  stereo/          (cameraL → namespace: stereo/)
        camera name: left            (cameraL → name: left)
        suffix:     /ImageData       (AMBF convention for image topics)

Authors: CASA CV Team
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import time
import threading

import cv2
import numpy as np

# --- ROS2 imports ---
# rclpy is the ROS2 Python client library. It allows us to:
#   - Create ROS2 nodes (named processes that communicate over topics)
#   - Subscribe to topics (receive data published by other nodes/simulators)
#   - Publish to topics (send data to other nodes)
#
# In ROS2, all communication happens through a "graph" of nodes. Each node
# has a unique name and can publish or subscribe to named data channels
# called "topics". The middleware (DDS) handles all the networking.
import rclpy
from rclpy.node import Node

# sensor_msgs.msg.Image is the standard ROS2 message type for camera images.
# AMBF publishes its camera frames using this exact message type.
# The message contains:
#   - header: timestamp and frame_id
#   - height, width: image dimensions
#   - encoding: pixel format string (e.g. 'rgb8', 'bgr8')
#   - data: raw pixel bytes as a flat array
from sensor_msgs.msg import Image

# CvBridge is the bridge between ROS2 Image messages and OpenCV/NumPy images.
# ROS stores images in its own serialized format; CvBridge converts them
# to a standard NumPy array (height x width x channels) that OpenCV can work with.
from cv_bridge import CvBridge, CvBridgeError


# ==============================================================================
# CONFIGURATION — edit these to change behavior
# ==============================================================================

# ROS2 topic where AMBF publishes the left (mono) camera frames.
# Defined in: ADF/world/world_stereo.yaml → cameraL → namespace: stereo/ → name: left
# Full topic = /ambf/env/ + stereo/ + left + /ImageData
CAMERA_TOPIC = "/ambf/env/stereo/left/ImageData"

# Image dimensions published by AMBF (set in world_stereo.yaml line 50)
IMG_WIDTH  = 640
IMG_HEIGHT = 480

# Number of red dot waypoints to place along the predicted needle path
NUM_WAYPOINTS = 8

# Size (radius in pixels) of each red dot drawn on the output image
DOT_RADIUS = 6

# Directory where annotated output images will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# If True, also display the result in an OpenCV window (requires a display / X server)
SHOW_WINDOW = True


# ==============================================================================
# STEP 3 — TISSUE SEGMENTATION (OpenCV HSV Color-Based)
# ==============================================================================
# This is a PLACEHOLDER segmentation using HSV color filtering.
# It works by isolating pixels that fall within certain color ranges
# that correspond to the phantom tissue in the AMBF simulation.
#
# >>> WHEN YOU HAVE A TRAINED U-NET: <<<
# Replace this class with a PyTorch-based SegmentationPipeline that loads
# your trained .pth weights. The interface stays the same — just make sure
# the replacement class has a segment(bgr_frame) method that returns a
# binary mask of shape (H, W) with values 0 or 255.
# ==============================================================================

class SegmentationPipeline:
    """
    OpenCV-based tissue segmentation using HSV color thresholding.

    How it works:
        1. Convert BGR image → HSV color space
           (HSV separates color (Hue) from brightness (Value), making it
           much easier to isolate colored objects under varying lighting)
        2. Apply multiple color range masks to capture tissue-like colors
           (pinkish/reddish tones of the phantom, plus any other tissue colors)
        3. Clean up the mask using morphological operations:
           - Opening (erode→dilate): removes small noise dots
           - Closing (dilate→erode): fills small holes inside the tissue region
        4. Return a clean binary mask (0 = background, 255 = tissue)

    Why HSV instead of U-Net right now:
        - No PyTorch dependency (saves 3GB+ install)
        - Actually produces meaningful results on the AMBF simulation
          (an untrained U-Net would just output random noise)
        - Easy to tune by adjusting the HSV ranges below
    """

    def __init__(self):
        # HSV range(s) for tissue-like colors in the AMBF simulation.
        # HSV ranges: H=[0,179], S=[0,255], V=[0,255] in OpenCV.
        #
        # These ranges target pinkish/reddish/brownish tones typical of
        # surgical phantoms. You may need to tune these after seeing the
        # actual simulator output — run the script, look at the raw frame,
        # and adjust ranges accordingly.
        #
        # Tip: Use this to find good ranges interactively:
        #   import cv2; cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
        self.hsv_ranges = [
            # Range 1: White / beige surgical phantom pads (like SIMPLE_LND_420006)
            {"lower": np.array([0, 0, 200]),     "upper": np.array([180, 80, 255])},
            # Range 2: Reddish-pink tones (low hue red wraps around 0-10)
            {"lower": np.array([0, 30, 40]),     "upper": np.array([15, 255, 255])},
            # Range 3: Reddish-pink tones (high hue red wraps around 160-179)
            {"lower": np.array([160, 30, 40]),    "upper": np.array([179, 255, 255])},
            # Range 4: Brownish / tissue tones
            {"lower": np.array([10, 30, 40]),     "upper": np.array([25, 255, 200])},
            # Range 5: Pinkish / lighter tissue
            {"lower": np.array([140, 20, 50]),    "upper": np.array([170, 255, 255])},
        ]

        # Morphological kernel for cleanup operations
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Segment tissue from a BGR camera frame using HSV color filtering.

        Args:
            bgr_frame: NumPy array, shape (H, W, 3), dtype uint8, BGR color.

        Returns:
            Binary mask: NumPy array, shape (H, W), dtype uint8.
            Values are 0 (background) or 255 (tissue).
        """
        # Convert BGR → HSV color space
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # Combine all HSV range masks with bitwise OR
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in self.hsv_ranges:
            mask = cv2.inRange(hsv, r["lower"], r["upper"])
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological opening: remove small noise specks
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel)

        # Morphological closing: fill small holes inside tissue regions
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        return combined_mask


# ==============================================================================
# STEP 4 — KEYPOINT / WAYPOINT PREDICTOR
# ==============================================================================

class NeedlePathPredictor:
    """
    Given a binary segmentation mask (tissue/wound region), predicts a set of
    ordered waypoints representing needle entry/exit points for suturing.

    HOW REAL SUTURING WORKS:
        A surgeon places stitches in a ZIGZAG pattern across the wound:
            Left side  ●───────● Right side
                        \     /
            Left side  ●───────● Right side
                        \     /
            Left side  ●───────● Right side

        The needle enters on one side of the wound, passes through tissue,
        exits on the other side, then moves along the wound and repeats.

    STRATEGY:
        1. Find the wound contour from the segmentation mask
        2. Compute the wound's long axis (principal direction) using PCA
        3. Compute the perpendicular axis (across the wound)
        4. Place waypoints along the wound, ALTERNATING between left and
           right sides of the wound center line
        5. Add controlled randomness to spacing and offset so each run
           produces different (but realistic) stitch patterns

    WHY RANDOMIZATION?
        These annotated images become training data for the imitation
        learning model. If every image has identical dot placement, the
        model will overfit to one pattern. Random variation teaches it
        to generalize to different suture spacings and positions.

    >>> FUTURE IMPROVEMENT: <<<
    Replace this with a trained keypoint regression model (e.g. a heatmap
    CNN) that predicts medically accurate needle entry/exit points based
    on the tissue geometry and doctor-labeled training data.
    """

    def __init__(self, num_waypoints: int = NUM_WAYPOINTS):
        # Must be even so we get pairs (one left, one right per stitch)
        self.num_waypoints = num_waypoints if num_waypoints % 2 == 0 else num_waypoints + 1
        # Number of stitch pairs along the wound
        self.num_stitches = self.num_waypoints // 2

    def predict(self, mask: np.ndarray) -> list:
        """
        Predict needle path waypoints on alternating sides of the wound.

        Args:
            mask: Binary NumPy array of shape (H, W), dtype uint8.
                  Pixel = 255 means tissue; 0 means background.

        Returns:
            List of (x, y) pixel coordinates in suturing order:
            [left_1, right_1, left_2, right_2, ...] — zigzag pattern.
            Returns empty list if no tissue is detected.
        """
        # --- Find tissue contour ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[WARNING] No contours found in mask. Check segmentation.")
            return []

        # Use the largest contour (main wound/tissue region)
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 100:
            print("[WARNING] Tissue region too small. Check segmentation.")
            return []

        # --- Get all tissue pixel coordinates ---
        tissue_pixels = np.column_stack(np.where(mask > 127))  # [row, col]

        if len(tissue_pixels) < 10:
            print("[WARNING] Very little tissue detected.")
            return []

        # Convert (row, col) → (x, y)
        points = tissue_pixels[:, ::-1].astype(np.float32)  # [x, y]

        # --- PCA: find wound direction and perpendicular ---
        mean = points.mean(axis=0)
        centered = points - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)

        # Principal axis = along the wound (direction of max variance)
        wound_axis = Vt[0]
        # Perpendicular axis = across the wound (left-right of wound)
        perp_axis = Vt[1]

        # --- Calculate wound extent along the principal axis ---
        projections_along = centered @ wound_axis
        t_min, t_max = projections_along.min(), projections_along.max()

        # Shrink slightly so dots aren't at the very edge of the tissue
        margin = 0.10  # 10% margin on each end
        t_range = t_max - t_min
        t_min += t_range * margin
        t_max -= t_range * margin

        # --- Calculate wound width (for offset distance) ---
        projections_perp = centered @ perp_axis
        wound_half_width = np.percentile(np.abs(projections_perp), 75)
        # Offset distance: how far left/right of center the dots go
        # Typically 40-70% of the wound half-width
        base_offset = wound_half_width * 0.55

        # --- Place waypoints with randomization ---
        rng = np.random.default_rng()  # fresh random seed each run

        # Random spacing: instead of perfectly equal spacing, add jitter
        # This creates more natural-looking stitch placement
        stitch_positions = np.linspace(t_min, t_max, self.num_stitches)
        spacing_jitter = (t_range / self.num_stitches) * 0.15  # ±15% jitter
        stitch_positions += rng.uniform(-spacing_jitter, spacing_jitter,
                                         size=self.num_stitches)

        waypoints = []
        for t in stitch_positions:
            # Center point along the wound axis
            center_pt = mean + t * wound_axis

            # Random offset variation: ±20% of base offset
            offset_variation = rng.uniform(0.80, 1.20)
            offset = base_offset * offset_variation

            # Left side point (negative perpendicular direction)
            left_pt = center_pt - offset * perp_axis
            lx = int(np.clip(round(left_pt[0]), 0, IMG_WIDTH - 1))
            ly = int(np.clip(round(left_pt[1]), 0, IMG_HEIGHT - 1))

            # Right side point (positive perpendicular direction)
            right_pt = center_pt + offset * perp_axis
            rx = int(np.clip(round(right_pt[0]), 0, IMG_WIDTH - 1))
            ry = int(np.clip(round(right_pt[1]), 0, IMG_HEIGHT - 1))

            # Suture order: enter left → exit right (one stitch)
            waypoints.append((lx, ly))
            waypoints.append((rx, ry))

        return waypoints


# ==============================================================================
# STEP 1 & 2 — ROS2 CAMERA SUBSCRIBER
# ==============================================================================

class AmbfCameraSubscriber(Node):
    """
    ROS2 Node that subscribes to the AMBF left camera image topic.

    --- What is a ROS2 Node? ---
    A Node is a single process in the ROS2 ecosystem. Nodes communicate
    by publishing or subscribing to 'topics' — named data channels.
    Every node has a unique name (ours is "needle_path_annotator").
    You can see all active nodes by running:  ros2 node list

    --- What is a Topic? ---
    A topic is a named channel through which nodes send/receive messages.
    AMBF acts as a *publisher* on /ambf/env/stereo/left/ImageData,
    continuously sending camera frames. Our Node *subscribes* to that topic
    and gets called back every time a new frame arrives.
    You can see all active topics by running:  ros2 topic list

    --- What is a Callback? ---
    When a new message arrives on the subscribed topic, ROS2 automatically
    calls our `image_callback` method with the message as argument. This is
    event-driven — we don't poll, we just wait and react.

    --- What is CvBridge? ---
    AMBF publishes images in ROS2's sensor_msgs/Image format, which is a
    binary blob with metadata (encoding, width, height, timestamp). CvBridge
    converts this into a standard OpenCV/NumPy array so we can process it.

    --- What is QoS (Quality of Service)? ---
    QoS controls how messages are delivered. The '10' in our subscriber
    means we keep a buffer of up to 10 messages. If we're too slow to
    process them, older messages get dropped. This is fine for our use case
    since we only need the latest frame.

    --- What is spin()? ---
    rclpy.spin(node) runs an infinite loop that listens for incoming
    messages and calls our callbacks. We run it in a background thread
    so our main code doesn't block.
    """

    def __init__(self):
        # Initialize this node with a unique name in the ROS2 graph.
        # Other nodes and tools (like `ros2 topic list`) will see this name.
        super().__init__("needle_path_annotator")

        # CvBridge instance — reused for every frame conversion
        self.bridge = CvBridge()

        # Storage for the latest received frame (NumPy BGR array)
        self.latest_frame = None

        # Thread lock to safely share latest_frame between the ROS2
        # spin thread and the main processing thread.
        # Without this lock, the main thread could read a half-written
        # frame while the callback is writing a new one → corrupted data.
        self._lock = threading.Lock()

        # --- Create the subscriber ---
        # This tells ROS2: "whenever a message of type Image arrives on
        # CAMERA_TOPIC, call self.image_callback with that message."
        #
        # Arguments:
        #   msg_type   : sensor_msgs.msg.Image — the ROS2 message type we expect
        #   topic      : CAMERA_TOPIC — the topic name to listen on
        #   callback   : self.image_callback — function called on each new message
        #   qos_profile: 10 — queue depth (buffer up to 10 messages)
        self.subscription = self.create_subscription(
            msg_type=Image,
            topic=CAMERA_TOPIC,
            callback=self.image_callback,
            qos_profile=10,
        )

        self.get_logger().info(
            f"[STEP 1] Subscribed to camera topic: {CAMERA_TOPIC}"
        )

    def image_callback(self, msg: Image) -> None:
        """
        Called automatically by ROS2 every time a new image frame arrives
        on the subscribed topic.

        --- ROS2 Image Message Fields ---
        msg.header.stamp   : timestamp when the image was captured
        msg.height         : image height in pixels (480)
        msg.width          : image width in pixels (640)
        msg.encoding       : pixel encoding string e.g. 'rgb8', 'bgr8', 'mono8'
        msg.data           : raw pixel bytes (flattened array)

        CvBridge.imgmsg_to_cv2(msg, 'bgr8') unpacks all of that into a
        standard NumPy array of shape (480, 640, 3) with BGR channel order
        — which is what OpenCV expects by default.
        """
        try:
            # Convert ROS2 Image message → OpenCV NumPy array (BGR, uint8)
            # 'bgr8' = Blue-Green-Red, 8 bits per channel
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            with self._lock:
                self.latest_frame = cv_image.copy()

        except CvBridgeError as e:
            self.get_logger().error(f"[CvBridge Error] {e}")

    def get_frame(self):
        """
        Thread-safe retrieval of the most recently received camera frame.

        Returns:
            NumPy array of shape (480, 640, 3), BGR uint8, or None if no
            frame has been received yet.
        """
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


# ==============================================================================
# STEP 5 — ANNOTATION & SAVING
# ==============================================================================

def annotate_and_save(
    original_frame: np.ndarray,
    mask: np.ndarray,
    waypoints: list,
    output_path: str,
) -> np.ndarray:
    """
    Draw the segmentation mask overlay and red dot waypoints on the
    original image, then save it to disk.

    Args:
        original_frame: BGR NumPy array (H, W, 3) — the raw camera frame.
        mask          : Binary NumPy array (H, W) — tissue segmentation mask.
        waypoints     : List of (x, y) pixel coords — predicted needle path.
        output_path   : Full path where the annotated image will be saved.

    Returns:
        The annotated image as a NumPy array.
    """
    annotated = original_frame.copy()

    # --- Draw segmentation mask overlay (semi-transparent green) ---
    mask_overlay = np.zeros_like(annotated)
    mask_overlay[mask > 127] = [0, 200, 0]  # green tint on tissue pixels

    # Blend: 70% original + 30% green mask overlay
    annotated = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)

    # --- Draw needle path waypoints (red dots) ---
    for i, (x, y) in enumerate(waypoints):
        # Draw filled red circle
        cv2.circle(annotated, center=(x, y), radius=DOT_RADIUS,
                   color=(0, 0, 255),   # BGR: pure red
                   thickness=-1)        # -1 = filled circle

        # Draw white outline for visibility against any background
        cv2.circle(annotated, center=(x, y), radius=DOT_RADIUS + 1,
                   color=(255, 255, 255),
                   thickness=1)

        # Number each dot so the path ordering is visible
        cv2.putText(
            annotated, str(i + 1),
            (x + DOT_RADIUS + 2, y + DOT_RADIUS // 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # --- Draw connecting line between waypoints (needle path visualization) ---
    if len(waypoints) > 1:
        pts = np.array(waypoints, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=False,
                      color=(0, 100, 255),  # orange-red line
                      thickness=1,
                      lineType=cv2.LINE_AA)

    # --- Add info text overlay ---
    cv2.putText(
        annotated,
        f"Waypoints: {len(waypoints)} | Tissue pixels: {np.sum(mask > 127)}",
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 255, 255),  # cyan text
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    # --- Save to disk ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"[STEP 5] Saved annotated image → {output_path}")

    return annotated


# ==============================================================================
# MAIN PIPELINE — ties all steps together
# ==============================================================================

def main():
    """
    Main entry point. Runs the full pipeline once:
        STEP 1: Initialize ROS2 and start the camera subscriber
        STEP 2: Wait for a valid frame from AMBF
        STEP 3: Run tissue segmentation on the frame
        STEP 4: Predict needle path waypoints from the segmentation mask
        STEP 5: Annotate and save the output image
    """

    print("=" * 70)
    print("  CASA CV Pipeline — Needle Path Annotator")
    print("=" * 70)

    # ==========================================================================
    # STEP 1: Initialize ROS2
    # ==========================================================================
    # rclpy.init() must be called exactly ONCE before creating any ROS2 nodes.
    # It sets up the underlying ROS2 middleware (DDS = Data Distribution Service).
    # DDS handles all the networking — discovering other nodes, serializing
    # messages, and transporting data over UDP/shared memory.
    print("\n[STEP 1] Initializing ROS2...")
    rclpy.init()

    # Create our camera subscriber node
    camera_node = AmbfCameraSubscriber()

    # Spin the node in a SEPARATE THREAD so ROS2 callbacks keep running
    # while our main thread does processing.
    #
    # Without threading: rclpy.spin() blocks forever → main code never runs.
    # With threading: spin runs in background, main thread continues below.
    #
    # daemon=True means this thread dies automatically when the main program
    # exits — no need to manually join/stop it.
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(camera_node,),
        daemon=True,
    )
    spin_thread.start()
    print("[STEP 1] ROS2 node started. Waiting for AMBF camera frames...")

    # ==========================================================================
    # STEP 2: Wait for a valid frame from AMBF
    # ==========================================================================
    # AMBF may take a moment to start publishing after the simulator launches.
    # We poll get_frame() until we receive at least one non-None frame.
    print("\n[STEP 2] Waiting for camera frame from AMBF simulator...")
    timeout_seconds = 30
    frame = None
    start_time = time.time()

    while frame is None:
        frame = camera_node.get_frame()
        if frame is None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n[ERROR] No frame received after {timeout_seconds}s.")
                print("        Is the AMBF simulator running?")
                print(f"        Expected topic: {CAMERA_TOPIC}")
                print("        Check with: ros2 topic list")
                rclpy.shutdown()
                sys.exit(1)
            print(f"  ... waiting for frame ({elapsed:.0f}s / {timeout_seconds}s)")
            time.sleep(1.0)

    print(f"[STEP 2] ✅ Frame received! Shape: {frame.shape}, dtype: {frame.dtype}")

    # Save the raw frame too (useful for debugging and comparison)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTPUT_DIR, f"raw_frame_{timestamp}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(raw_path, frame)
    print(f"[STEP 2] Raw frame saved → {raw_path}")

    # ==========================================================================
    # STEP 3: Run Tissue Segmentation
    # ==========================================================================
    # Currently using OpenCV HSV color-based segmentation as a placeholder.
    # Replace with a trained U-Net model when weights are available.
    print("\n[STEP 3] Running tissue segmentation (HSV color filtering)...")
    segmenter = SegmentationPipeline()
    tissue_mask = segmenter.segment(frame)

    tissue_pixel_count = int(np.sum(tissue_mask > 127))
    total_pixels = tissue_mask.size
    tissue_pct = 100.0 * tissue_pixel_count / total_pixels
    print(f"[STEP 3] Tissue detected: {tissue_pixel_count:,} / {total_pixels:,} pixels "
          f"({tissue_pct:.1f}%)")

    # Save the mask too for debugging
    mask_path = os.path.join(OUTPUT_DIR, f"tissue_mask_{timestamp}.png")
    cv2.imwrite(mask_path, tissue_mask)
    print(f"[STEP 3] Tissue mask saved → {mask_path}")

    # ==========================================================================
    # STEP 4: Predict Needle Path Waypoints
    # ==========================================================================
    print(f"\n[STEP 4] Predicting needle path ({NUM_WAYPOINTS} waypoints)...")
    predictor = NeedlePathPredictor(num_waypoints=NUM_WAYPOINTS)
    waypoints = predictor.predict(tissue_mask)

    if waypoints:
        print(f"[STEP 4] ✅ Waypoints predicted:")
        for i, (x, y) in enumerate(waypoints):
            print(f"         [{i+1}] x={x:4d}, y={y:4d}")
    else:
        print("[STEP 4] ⚠ No waypoints generated (tissue not detected).")
        print("         Try adjusting HSV ranges in SegmentationPipeline.__init__()")

    # ==========================================================================
    # STEP 5: Annotate and Save
    # ==========================================================================
    print("\n[STEP 5] Annotating image and saving...")
    output_path = os.path.join(OUTPUT_DIR, f"needle_path_{timestamp}.png")
    annotated_image = annotate_and_save(frame, tissue_mask, waypoints, output_path)

    # Optionally display in a window (requires a display / X11 server)
    if SHOW_WINDOW:
        try:
            cv2.imshow("CASA — Needle Path Annotator", annotated_image)
            print("\n[STEP 5] Press any key in the image window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("[STEP 5] Could not open display window (no X server / headless mode).")

    # ==========================================================================
    # CLEANUP
    # ==========================================================================
    # rclpy.shutdown() cleanly terminates the ROS2 context and middleware.
    # Always call this before exiting if you called rclpy.init().
    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    print(f"  Raw frame  → {raw_path}")
    print(f"  Mask       → {mask_path}")
    print(f"  Annotated  → {output_path}")
    print("=" * 70)

    rclpy.shutdown()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
