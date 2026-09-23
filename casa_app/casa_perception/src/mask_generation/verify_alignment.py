#!/usr/bin/env python3
"""
================================================================================
  verify_alignment.py -- overlay generated masks on their source images
================================================================================

PURPOSE
-------
generate_mask_openCV.py reads images from:

    <BASE_DATASET_DIR>/<run>/images/

and writes masks (for frames that passed validation) to:

    <BASE_DATASET_DIR>/<run>/mask_openCV/<name>_mask.jpg

This script pairs each saved mask back up with its original image, tints
the mask region green, and writes the blended result to a new folder
sitting right next to the other two:

    <BASE_DATASET_DIR>/<run>/alignment_check/<name>_alignment.jpg

Anything that can't be paired (missing original, missing mask, size
mismatch, unreadable file) is skipped and logged to
`_ALIGNMENT_SKIPPED.txt` in that same folder, mirroring the review-log
pattern used by generate_mask_openCV.py, instead of silently dropping it.

HOW TO RUN
----------
    python3 verify_alignment.py
"""

import cv2
import numpy as np
import os
import glob

# ==============================================================================
# CONFIGURATION -- kept identical to generate_mask_openCV.py so the two
# scripts always point at the same data. If you change paths/runs there,
# change them here too.
# ==============================================================================

BASE_DATASET_DIR = "../../dataset/processed"
RUNS = ["run_01", "run_02", "run_03", "run_04"]

IMAGE_SUBDIR = "images"
MASK_SUBDIR = "mask_openCV"
OUTPUT_SUBDIR = "alignment_check"

MASK_SUFFIX = "_mask"          # e.g. frame001_mask.jpg
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# --- overlay appearance ---
FILL_COLOR = (0, 255, 0)       # BGR green -- semi-transparent fill over the wound
FILL_ALPHA = 0.45              # opacity of the fill (0 = invisible, 1 = solid)


# ==============================================================================
# HELPERS
# ==============================================================================

def find_original_image(images_dir, base_name):
    """Return the path to the original image matching base_name, trying
    each supported extension, or None if none exist."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def build_overlay(original, mask):
    """Blend a semi-transparent green fill of `mask` onto `original`.
    Returns the blended BGR image."""
    binary_mask = (mask == 255)

    tinted = original.copy()
    tinted[binary_mask] = FILL_COLOR
    blended = cv2.addWeighted(original, 1 - FILL_ALPHA, tinted, FILL_ALPHA, 0)

    return blended


# ==============================================================================
# PER-RUN PIPELINE
# ==============================================================================

def process_run(run):
    images_dir = os.path.join(BASE_DATASET_DIR, run, IMAGE_SUBDIR)
    masks_dir = os.path.join(BASE_DATASET_DIR, run, MASK_SUBDIR)
    output_dir = os.path.join(BASE_DATASET_DIR, run, OUTPUT_SUBDIR)

    if not os.path.isdir(masks_dir):
        print(f"[{run}] No mask folder at {masks_dir}. Skipping.")
        return

    mask_files = sorted(glob.glob(os.path.join(masks_dir, f"*{MASK_SUFFIX}.jpg")))
    if not mask_files:
        print(f"[{run}] No masks found in {masks_dir}. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[{run}] Found {len(mask_files)} mask(s). Saving overlays to {output_dir}/")

    skipped = []
    saved_count = 0

    for mask_path in mask_files:
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]  # e.g. frame001_mask
        if not mask_filename.endswith(MASK_SUFFIX):
            skipped.append(f"{mask_filename}: unexpected mask filename (no '{MASK_SUFFIX}' suffix)")
            continue
        base_name = mask_filename[: -len(MASK_SUFFIX)]

        original_path = find_original_image(images_dir, base_name)
        if original_path is None:
            skipped.append(f"{base_name}: no matching original image in {images_dir}")
            continue

        original = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if original is None:
            skipped.append(f"{base_name}: could not load original image ({original_path})")
            continue
        if mask is None:
            skipped.append(f"{base_name}: could not load mask ({mask_path})")
            continue
        if original.shape[:2] != mask.shape[:2]:
            skipped.append(
                f"{base_name}: size mismatch (original {original.shape[:2]} vs mask {mask.shape[:2]})"
            )
            continue

        overlay = build_overlay(original, mask)
        output_path = os.path.join(output_dir, f"{base_name}_alignment.jpg")
        cv2.imwrite(output_path, overlay)
        saved_count += 1

    print(f"  {saved_count}/{len(mask_files)} overlays saved. {len(skipped)} skipped.")

    if skipped:
        skip_log_path = os.path.join(output_dir, "_ALIGNMENT_SKIPPED.txt")
        with open(skip_log_path, "w") as f:
            f.write(
                f"{len(skipped)} of {len(mask_files)} masks could not be paired with an "
                f"original image and were skipped:\n\n"
            )
            f.write("\n".join(skipped))
        print(f"  See {skip_log_path} for the list.")


# ==============================================================================
# BATCH DRIVER
# ==============================================================================

def process_batch():
    for run in RUNS:
        process_run(run)
    print("\nAlignment check complete.")


if __name__ == "__main__":
    process_batch()