#!/usr/bin/env python3
"""
visualize_masks.py
==================
Visual QA tool: overlays generated masks on their corresponding raw frames
to verify spatial alignment.

USAGE
-----
    python3 visualize_masks.py [--run-dirs RUN1 RUN2 ...]
                               [--frames N [M ...]]
                               [--out-dir PATH]
                               [--every N]

    --run-dirs     One or more run directories under dataset/processed/.
                   Defaults to all (run_01, run_02, run_03, run_04).
    --frames       Specific frame indices to include (0-indexed). Default: 0 6.
    --every        Instead of --frames, sample every N-th frame.
    --out-dir      Where to save the overlay images. Default: ./mask_qa/

WHAT IT PRODUCES
----------------
For each (run, frame) pair:
    <out-dir>/<run>_frame_NNNN.png
        Left half:  raw RGB frame with the mask overlaid as a semi-transparent
                    red tint + white contour.
        Right half: raw frame + mask as pure green outline only (easier for
                    alignment inspection).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


DATASET_RAW  = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../dataset/raw"
))
DATASET_PROC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../dataset/processed"
))


def load_all_runs(processed_dir):
    """Return sorted list of run names that have a 'masks/' sub-folder."""
    runs = []
    for name in sorted(os.listdir(processed_dir)):
        mask_path = os.path.join(processed_dir, name, "masks")
        if os.path.isdir(mask_path):
            runs.append(name)
    return runs


def overlay_mask_on_frame(raw_bgr, mask_gray):
    """Return two visualisations of the mask on the raw frame.

    Returns
    -------
    tinted : ndarray  -- red tint where mask is non-zero + white contour
    outline: ndarray  -- green contour only (no tint)
    """
    tinted = raw_bgr.copy()
    where = mask_gray > 0

    # Semi-transparent red tint
    tinted[where] = tinted[where] * 0.4 + np.array([0, 0, 200]) * 0.6

    # White contour on tinted
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tinted, contours, -1, (255, 255, 255), 1)

    # Green contour only
    outline = raw_bgr.copy()
    cv2.drawContours(outline, contours, -1, (0, 255, 0), 1)

    return tinted, outline


def process_frame(run_dir_raw, run_dir_proc, frame_idx, out_dir, run_name):
    raw_path  = os.path.join(run_dir_raw,  f"frame_{frame_idx:04d}.png")
    mask_path = os.path.join(run_dir_proc, "masks", f"frame_{frame_idx:04d}.png")

    if not os.path.isfile(raw_path):
        print(f"  [SKIP] raw not found:  {raw_path}")
        return False
    if not os.path.isfile(mask_path):
        print(f"  [SKIP] mask not found: {mask_path}")
        return False

    raw  = cv2.imread(raw_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if raw is None or mask is None:
        print(f"  [SKIP] could not read images for frame {frame_idx:04d}")
        return False

    # Resize mask to match raw if needed
    if mask.shape[:2] != raw.shape[:2]:
        mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    tinted, outline = overlay_mask_on_frame(raw, mask)

    # Add labels
    def label(img, text):
        cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)

    label(tinted,  f"{run_name} | frame_{frame_idx:04d} | mask (tinted+contour)")
    label(outline, f"{run_name} | frame_{frame_idx:04d} | mask (outline only)")

    composite = np.hstack([tinted, outline])
    out_path  = os.path.join(out_dir, f"{run_name}_frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, composite)
    print(f"  Saved: {out_path}")

    # Print quick stats
    n_white = np.count_nonzero(mask)
    h, w = mask.shape[:2]
    print(f"    mask coverage: {n_white}/{h*w} px ({100*n_white/(h*w):.2f}%)")
    if n_white > 0:
        ys, xs = np.where(mask > 0)
        print(f"    mask bbox:     x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]")
        print(f"    mask centroid: ({xs.mean():.0f}, {ys.mean():.0f})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    proc_dir = os.path.normpath(DATASET_PROC)
    raw_dir  = os.path.normpath(DATASET_RAW)

    parser.add_argument("--run-dirs", nargs="+", default=None,
                        help="Run names (e.g. run_01 run_02). Default: all found.")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 6],
                        help="Frame indices to visualise. Default: 0 6.")
    parser.add_argument("--every", type=int, default=None,
                        help="Sample every N-th frame instead of --frames.")
    parser.add_argument("--out-dir", default="./mask_qa",
                        help="Output folder. Default: ./mask_qa/")
    parser.add_argument("--processed-dir", default=proc_dir)
    parser.add_argument("--raw-dir", default=raw_dir)
    args = parser.parse_args()

    if not os.path.isdir(args.processed_dir):
        print(f"ERROR: processed dir not found: {args.processed_dir}")
        sys.exit(1)

    runs = args.run_dirs if args.run_dirs else load_all_runs(args.processed_dir)
    if not runs:
        print(f"ERROR: no run folders with masks/ found in {args.processed_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    for run in runs:
        run_dir_raw  = os.path.join(args.raw_dir,  run)
        run_dir_proc = os.path.join(args.processed_dir, run)

        print(f"\n=== {run} ===")

        if args.every is not None:
            # Discover all available mask frames
            mask_files = sorted(glob.glob(
                os.path.join(run_dir_proc, "masks", "frame_*.png")
            ))
            frame_ids = []
            for mf in mask_files:
                base = os.path.basename(mf)          # frame_NNNN.png
                try:
                    frame_ids.append(int(base[6:10]))
                except ValueError:
                    pass
            frame_ids = frame_ids[::args.every]
        else:
            frame_ids = args.frames

        for fid in frame_ids:
            process_frame(run_dir_raw, run_dir_proc, fid, args.out_dir, run)

    print(f"\n[DONE] Overlays saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
