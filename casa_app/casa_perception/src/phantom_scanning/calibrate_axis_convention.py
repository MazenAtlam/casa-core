#!/usr/bin/env python3
"""
calibrate_axis_convention.py
============================
Companion script for ``orbit_scan.py --calibrate``.

PURPOSE
-------
After running ``python3 orbit_scan.py --calibrate``, this script reads the
saved calibration frames + poses JSON and empirically determines:

  1. Which local camera axis (local_X or local_Y) maps to the image u-axis
     (horizontal, left→right).
  2. Which local camera axis (local_X or local_Y) maps to the image v-axis
     (vertical, top→bottom).
  3. The sign of each mapping (positive or negative).

HOW IT WORKS
------------
For each calibration frame the script:

  a) Reads the commanded camera pos & rpy.
  b) Computes R_cam = euler_to_matrix(roll, pitch, yaw).
  c) Projects the phantom centre and two landmark offsets (+X, +Y in world)
     into camera space: ``cam_local = R_cam.T @ (world_pt - t_cam)``.
  d) Looks at each candidate mapping (u = ±local[0]/depth or ±local[1]/depth)
     and checks which one produces a u-value that matches the visual centre of
     the PINK phantom blob in the captured image.

The mapping that scores best across ALL calibration frames is printed as the
correct convention and a ready-to-paste code snippet for
``generate_masks_projection.py`` is shown.

USAGE
-----
    python3 calibrate_axis_convention.py [--calib-dir PATH]

    --calib-dir   Path to the folder produced by orbit_scan.py --calibrate.
                  Default: ./orbit_calibration
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def euler_to_matrix(roll, pitch, yaw):
    """Build 3x3 rotation matrix from roll/pitch/yaw (AMBF ZYX extrinsic)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def detect_phantom_centroid(img_bgr, hue_lo=290, hue_hi=320, sat_lo=100, val_lo=100):
    """Return (cx, cy) pixel centroid of the largest PINK region in an image.

    Uses HSV thresholding.  Pink hue ≈ 300–330° (in OpenCV 0–180 scale ≈ 150–165).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # HSV pink: hue in [145, 175], high sat, high val
    lo = np.array([145, 60, 100])
    hi = np.array([175, 255, 255])
    mask = cv2.inRange(hsv, lo, hi)

    # Clean up
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    # Largest contour
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] < 1e-6:
        return None, mask
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy), mask


def project_point(world_pt, t_cam, R_cam, fx, fy, cx_img, cy_img,
                  u_axis, u_sign, v_axis, v_sign):
    """Project world_pt into pixel (u, v) under the given axis convention.

    u_axis, v_axis: 0 or 1 (indices into the 3D cam_local vector).
    u_sign, v_sign: +1 or -1.
    forward is always -cam_local[2].
    """
    cam_local = R_cam.T @ (world_pt - t_cam)
    depth = -cam_local[2]   # forward = local -Z
    if depth < 1e-6:
        return None, None
    u = fx * (u_sign * cam_local[u_axis] / depth) + cx_img
    v = fy * (v_sign * cam_local[v_axis] / depth) + cy_img
    return u, v


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_calibration")
    parser.add_argument("--calib-dir", default=here,
                        help=f"Calibration folder from orbit_scan.py --calibrate. Default: {here}")
    parser.add_argument("--fov-v", type=float, default=0.9599310886,
                        help="Vertical FOV in radians (default: 0.9599310886)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated debug images to calib_dir/annotated/.")
    args = parser.parse_args()

    calib_dir = args.calib_dir
    poses_path = os.path.join(calib_dir, "calibration_poses.json")

    if not os.path.isfile(poses_path):
        print(f"ERROR: {poses_path} not found.  Run orbit_scan.py --calibrate first.")
        sys.exit(1)

    with open(poses_path) as f:
        poses = json.load(f)

    # Camera intrinsics
    fy = (args.height / 2.0) / math.tan(args.fov_v / 2.0)
    fx = fy
    cx_img = args.width / 2.0
    cy_img = args.height / 2.0

    if args.save_annotated:
        ann_dir = os.path.join(calib_dir, "annotated")
        os.makedirs(ann_dir, exist_ok=True)

    # Candidate conventions: (u_axis_idx, u_sign, v_axis_idx, v_sign)
    candidates = []
    for u_ax in range(2):
        for u_s in [+1, -1]:
            for v_ax in range(2):
                if v_ax == u_ax:
                    continue   # u and v must use different axes
                for v_s in [+1, -1]:
                    candidates.append((u_ax, u_s, v_ax, v_s))

    # For each candidate accumulate total squared error (pred - measured)
    errors = {c: 0.0 for c in candidates}
    n_frames = 0

    for pose in poses:
        img_path = os.path.join(calib_dir, pose["image"])
        if not os.path.isfile(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] Could not read: {img_path}")
            continue

        t_cam = np.array([pose["camera_pos"]["x"],
                          pose["camera_pos"]["y"],
                          pose["camera_pos"]["z"]], dtype=np.float64)
        rpy = pose["camera_rpy"]
        R_cam = euler_to_matrix(rpy["roll"], rpy["pitch"], rpy["yaw"])

        ph = pose["phantom_position"]
        t_phantom = np.array([ph["x"], ph["y"], ph["z"]], dtype=np.float64)

        # Detect visible centroid of the pink phantom
        centroid, hsv_mask = detect_phantom_centroid(img)
        if centroid is None:
            print(f"  [SKIP] No pink phantom detected in {pose['image']} ({pose['label']})")
            continue

        measured_u, measured_v = centroid
        print(f"  [{pose['label']}] phantom centroid detected at pixel ({measured_u:.1f}, {measured_v:.1f})")

        # Score each candidate: project phantom centre, compare with detection
        for cand in candidates:
            u_ax, u_s, v_ax, v_s = cand
            pred_u, pred_v = project_point(
                t_phantom, t_cam, R_cam,
                fx, fy, cx_img, cy_img,
                u_ax, u_s, v_ax, v_s
            )
            if pred_u is None:
                errors[cand] += 1e9
                continue
            errors[cand] += (pred_u - measured_u) ** 2 + (pred_v - measured_v) ** 2

        if args.save_annotated:
            vis = img.copy()
            cx_p, cy_p = int(round(measured_u)), int(round(measured_v))
            cv2.drawMarker(vis, (cx_p, cy_p), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.imwrite(os.path.join(ann_dir, pose["image"]), vis)

        n_frames += 1

    if n_frames == 0:
        print("\n[ERROR] No valid frames could be processed.")
        sys.exit(1)

    # Sort by total error
    ranked = sorted(errors.items(), key=lambda kv: kv[1])
    best_cand, best_err = ranked[0]
    u_ax, u_s, v_ax, v_s = best_cand

    u_name = ("local_X" if u_ax == 0 else "local_Y")
    v_name = ("local_X" if v_ax == 0 else "local_Y")
    u_s_str = "+" if u_s > 0 else "-"
    v_s_str = "+" if v_s > 0 else "-"

    print("\n" + "=" * 65)
    print("CALIBRATION RESULT")
    print("=" * 65)
    print(f"  Best convention (RMSE per axis ≈ {math.sqrt(best_err/n_frames):.1f}px):")
    print(f"    u (image left→right) = {u_s_str}cam_local[{u_ax}]  ({u_s_str}{u_name})")
    print(f"    v (image top→bottom) = {v_s_str}cam_local[{v_ax}]  ({v_s_str}{v_name})")
    print()
    print("  Runner-up errors:")
    for cand, err in ranked[1:4]:
        rmse = math.sqrt(err / max(n_frames, 1))
        print(f"    local[{cand[0]}]*{cand[1]:+d} / local[{cand[2]}]*{cand[3]:+d}  RMSE≈{rmse:.1f}px")
    print()

    # --- Code snippet for generate_masks_projection.py ---
    # Current code uses:
    #   x_opt = RIGHT_SIGN * cam_local[:, 1]   (local_Y)
    #   y_opt = -UP_SIGN   * cam_local[:, 0]   (−local_X)
    # New code should use the best candidate.
    print("  Code snippet for generate_masks_projection.py:")
    print("  " + "-" * 55)
    if u_ax == 0:
        x_line = f"x_opt = RIGHT_SIGN * cam_local[:, 0]   # +local_X → u"
    else:
        x_line = f"x_opt = RIGHT_SIGN * cam_local[:, 1]   # +local_Y → u"
    if u_s < 0:
        x_line = x_line.replace("RIGHT_SIGN", "-RIGHT_SIGN")

    if v_ax == 0:
        y_line = f"y_opt = -UP_SIGN   * cam_local[:, 0]   # −local_X → v"
    else:
        y_line = f"y_opt = -UP_SIGN   * cam_local[:, 1]   # −local_Y → v"
    if v_s > 0:
        y_line = y_line.replace("-UP_SIGN", "UP_SIGN")

    print(f"    {x_line}")
    print(f"    {y_line}")
    print("    z_opt = -cam_local[:, 2]               # forward = −local_Z (unchanged)")
    print()
    print("  Apply this to generate_masks_projection.py and regenerate all masks.")
    print("=" * 65)


if __name__ == "__main__":
    main()
