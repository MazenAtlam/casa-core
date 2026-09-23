# #!/usr/bin/env python3
# """
# ================================================================================
#   generate_mask_openCV.py -- robust version
# ================================================================================

# WHAT WAS ACTUALLY WRONG (diagnosed against real project images, not guessed)
# -------------------------------------------------------------------------------
# Shrinking the final dilation kernel didn't help because the problem was
# never the dilation step -- it happens earlier, and dilation just inflates
# whatever shape came out of it, right or wrong. Two real, measured issues:

# 1. SCALE: the erode/close/dilate kernels were fixed pixel counts (25x25,
#    15x15, 5x5). Measured on real captures from this project: the wound's
#    detected area ranges from ~500px to ~22,500px depending on zoom (a 44x
#    range) -- a fixed-pixel kernel that's reasonable at one zoom level is
#    wildly wrong at another. Fix: kernel sizes are now computed as a
#    FRACTION of the detected phantom's own size in each image, so they
#    automatically scale with zoom instead of needing hand-tuning per shot.

# 2. NO VALIDATION: the script always takes "the largest contour" and trusts
#    it's the wound, with zero check on whether that's plausible. Measured
#    directly: on an image with the arm/needle occluding the wound, the
#    wound/phantom area ratio drops from ~9-10% to ~4.1% -- the tool breaks
#    up the wound's edges enough that the morphological closing step can't
#    connect them into the real wound shape anymore, and something else
#    (tool edge, thread, shadow, texture noise) can end up being "the
#    largest contour" instead. The old script would silently save that as
#    if it were a correct wound mask. This version checks the candidate
#    against what a real wound should look like (thin and elongated, a
#    plausible size relative to the phantom, roughly centered) and if it
#    fails, the mask is NOT saved blindly -- it's logged to a review list
#    instead, so you get a short list of frames to actually look at rather
#    than 2000 unreviewed files of unknown quality.

# HOW TO RUN
# ----------
#     python3 generate_mask_openCV.py

# OUTPUT
# ------
#     <run>/mask_openCV/<name>_mask.jpg      -- as before, for frames that
#                                               passed validation
#     <run>/mask_openCV/_REVIEW_NEEDED.txt   -- NEW: list of frames that
#                                               didn't produce a plausible
#                                               wound shape, with the reason
#                                               (too big / too small / not
#                                               elongated / no contour found)
# """

# import cv2
# import numpy as np
# import os
# import glob


# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================

# BASE_DATASET_DIR = "../../dataset/processed"
# RUNS = ["run_01", "run_02", "run_03", "run_04"]

# # --- phantom silhouette detection ---
# LOWER_PINK = np.array([130, 40, 40])
# UPPER_PINK = np.array([170, 255, 255])

# # --- scale-relative kernel sizes (fractions of the phantom's own bounding-box
# #     diagonal in THIS image -- replaces the old fixed-pixel kernels, which
# #     is what actually caused the inconsistency across zoom levels) ---
# ERODE_FRACTION = 0.035    # was fixed 25x25px
# CLOSE_FRACTION = 0.020    # was fixed 15x15px
# DILATE_FRACTION = 0.012   # was fixed 5x5px -- this is the margin-for-suturing step

# # --- Canny edge thresholds (unchanged -- not the diagnosed problem, leave
# #     alone unless review of the flagged images shows this needs tuning too) ---
# CANNY_LOW, CANNY_HIGH = 40, 120

# # --- plausibility validation for the candidate wound contour ---
# MIN_WOUND_AREA_FRACTION = 0.005   # < 0.5% of phantom area -- too small, probably noise
# MAX_WOUND_AREA_FRACTION = 0.25    # > 25% of phantom area -- too big, probably not the wound
# MIN_ELONGATION = 1.8              # PCA major/minor axis ratio -- a wound is a thin line,
#                                     # not a blob; same technique used elsewhere in this
#                                     # project for the same reason


# # ==============================================================================
# # HELPERS
# # ==============================================================================

# def get_phantom_silhouette(image):
#     """Returns (mask, contour, diagonal_px) for the phantom's own outline,
#     or (None, None, None) if nothing plausible was found."""
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     raw_mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)
#     contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, None, None

#     largest = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest)
#     diagonal = float(np.hypot(w, h))

#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
#     return mask, largest, diagonal


# def odd_kernel(size_px):
#     size_px = max(3, int(round(size_px)))
#     if size_px % 2 == 0:
#         size_px += 1
#     return size_px


# def contour_elongation(contour):
#     """PCA major/minor axis ratio of the contour's points -- high for a thin
#     line (the wound), low for a round/blob shape (likely not the wound)."""
#     pts = contour.reshape(-1, 2).astype(np.float32)
#     if len(pts) < 5:
#         return 0.0
#     mean, eigvecs, eigvals = cv2.PCACompute2(pts, mean=np.array([]))
#     eigvals = eigvals.flatten()
#     return float(np.sqrt(max(eigvals[0], 1e-6) / max(eigvals[1], 1e-6)))


# def validate_wound_contour(contour, phantom_area):
#     """Returns (is_valid, reason_if_not)."""
#     area = cv2.contourArea(contour)
#     if area <= 0:
#         return False, "zero area"
#     ratio = area / max(phantom_area, 1)
#     if ratio < MIN_WOUND_AREA_FRACTION:
#         return False, f"too small ({ratio*100:.2f}% of phantom, min {MIN_WOUND_AREA_FRACTION*100:.1f}%)"
#     if ratio > MAX_WOUND_AREA_FRACTION:
#         return False, f"too large ({ratio*100:.1f}% of phantom, max {MAX_WOUND_AREA_FRACTION*100:.0f}%)"
#     elong = contour_elongation(contour)
#     if elong < MIN_ELONGATION:
#         return False, f"not elongated enough ({elong:.2f}, min {MIN_ELONGATION})"
#     return True, ""


# # ==============================================================================
# # PER-IMAGE PIPELINE
# # ==============================================================================

# def generate_mask(image):
#     """
#     Returns (mask_or_None, status_string). mask is None if no plausible
#     wound contour was found -- caller should log this for review, not save
#     a guessed mask.
#     """
#     phantom_mask, phantom_contour, diagonal = get_phantom_silhouette(image)
#     if phantom_mask is None:
#         return None, "no phantom silhouette found"
#     phantom_area = cv2.contourArea(phantom_contour)

#     roi = cv2.bitwise_and(image, image, mask=phantom_mask)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(gray)
#     blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
#     edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

#     erode_k = odd_kernel(diagonal * ERODE_FRACTION)
#     inner_zone_mask = cv2.erode(phantom_mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)
#     internal_edges = cv2.bitwise_and(edges, edges, mask=inner_zone_mask)

#     close_k = odd_kernel(diagonal * CLOSE_FRACTION)
#     connected_edges = cv2.morphologyEx(internal_edges, cv2.MORPH_CLOSE,
#                                         np.ones((close_k, close_k), np.uint8))

#     wound_contours, _ = cv2.findContours(connected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not wound_contours:
#         return None, "no candidate contour after edge detection"

#     # Check candidates largest-first; use the first one that's actually
#     # plausible, rather than blindly trusting "largest = correct" like
#     # the old version did.
#     candidates = sorted(wound_contours, key=cv2.contourArea, reverse=True)
#     for c in candidates[:5]:   # only worth checking a handful of the biggest
#         valid, reason = validate_wound_contour(c, phantom_area)
#         if valid:
#             final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
#             cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
#             dilate_k = odd_kernel(diagonal * DILATE_FRACTION)
#             final_mask = cv2.dilate(final_mask, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)
#             return final_mask, "ok"

#     # Nothing plausible -- report why the best candidate failed, for the review log
#     _, reason = validate_wound_contour(candidates[0], phantom_area)
#     return None, f"largest candidate rejected: {reason}"


# # ==============================================================================
# # BATCH DRIVER (same structure as before)
# # ==============================================================================

# def process_batch():
#     for run in RUNS:
#         input_dir = os.path.join(BASE_DATASET_DIR, run, "images")
#         output_dir = os.path.join(BASE_DATASET_DIR, run, "mask_openCV")

#         image_files = []
#         for ext in ("*.jpeg", "*.jpg", "*.png"):
#             image_files.extend(glob.glob(os.path.join(input_dir, ext)))

#         if not image_files:
#             print(f"No images found in {input_dir}. Skipping {run}.")
#             continue

#         os.makedirs(output_dir, exist_ok=True)
#         print(f"\n[{run}] Found {len(image_files)} image(s). Saving masks to {output_dir}/")

#         review_log = []
#         saved_count = 0

#         for image_path in image_files:
#             base_name = os.path.splitext(os.path.basename(image_path))[0]
#             image = cv2.imread(image_path)
#             if image is None:
#                 review_log.append(f"{base_name}: could not load image")
#                 continue

#             mask, status = generate_mask(image)
#             if mask is not None:
#                 output_path = os.path.join(output_dir, f"{base_name}_mask.jpg")
#                 cv2.imwrite(output_path, mask)
#                 saved_count += 1
#             else:
#                 review_log.append(f"{base_name}: {status}")

#         print(f"  {saved_count}/{len(image_files)} masks saved. "
#               f"{len(review_log)} flagged for review.")

#         if review_log:
#             review_path = os.path.join(output_dir, "_REVIEW_NEEDED.txt")
#             with open(review_path, "w") as f:
#                 f.write(f"{len(review_log)} of {len(image_files)} images did not produce "
#                         f"a plausible wound mask and were skipped:\n\n")
#                 f.write("\n".join(review_log))
#             print(f"  See {review_path} for the list.")

#     print("\nBatch processing complete.")


# if __name__ == "__main__":
#     process_batch()


#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DATASET_DIR = "../../dataset/processed"
RUNS = ["run_01", "run_02", "run_03", "run_04"]

LOWER_PINK = np.array([130, 40, 40])
UPPER_PINK = np.array([170, 255, 255])

MAX_TOOL_SATURATION = 60
MAX_TOOL_VALUE = 40
TOOL_DILATE_FRACTION = 0.010

ERODE_FRACTION = 0.035
CLOSE_FRACTION = 0.020
DILATE_FRACTION = 0.012

CANNY_LOW, CANNY_HIGH = 40, 120

MIN_WOUND_AREA_FRACTION = 0.005
MAX_WOUND_AREA_FRACTION = 0.25
MIN_ELONGATION = 1.8

# ==============================================================================
# HELPERS
# ==============================================================================
def get_phantom_silhouette(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    diagonal = float(np.hypot(w, h))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask, largest, diagonal

def odd_kernel(size_px):
    size_px = max(3, int(round(size_px)))
    if size_px % 2 == 0:
        size_px += 1
    return size_px

def contour_elongation(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 0.0
    mean, eigvecs, eigvals = cv2.PCACompute2(pts, mean=np.array([]))
    eigvals = eigvals.flatten()
    return float(np.sqrt(max(eigvals[0], 1e-6) / max(eigvals[1], 1e-6)))

def validate_wound_contour(contour, phantom_area):
    area = cv2.contourArea(contour)
    if area <= 0:
        return False, "zero area"
    ratio = area / max(phantom_area, 1)
    if ratio < MIN_WOUND_AREA_FRACTION:
        return False, f"too small ({ratio*100:.2f}% of phantom, min {MIN_WOUND_AREA_FRACTION*100:.1f}%)"
    if ratio > MAX_WOUND_AREA_FRACTION:
        return False, f"too large ({ratio*100:.1f}% of phantom, max {MAX_WOUND_AREA_FRACTION*100:.0f}%)"
    elong = contour_elongation(contour)
    if elong < MIN_ELONGATION:
        return False, f"not elongated enough ({elong:.2f}, min {MIN_ELONGATION})"
    return True, ""

# ==============================================================================
# PER-IMAGE PIPELINE
# ==============================================================================
def generate_mask(image):
    phantom_mask, phantom_contour, diagonal = get_phantom_silhouette(image)
    if phantom_mask is None:
        return None, "no phantom silhouette found"
    phantom_area = cv2.contourArea(phantom_contour)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tool_sat_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, MAX_TOOL_SATURATION, 255]))
    tool_dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, MAX_TOOL_VALUE]))
    tool_mask = cv2.bitwise_or(tool_sat_mask, tool_dark_mask)

    tool_dilate_k = odd_kernel(diagonal * TOOL_DILATE_FRACTION)
    tool_mask = cv2.dilate(tool_mask, np.ones((tool_dilate_k, tool_dilate_k), np.uint8), iterations=1)

    roi = cv2.bitwise_and(image, image, mask=phantom_mask)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    erode_k = odd_kernel(diagonal * ERODE_FRACTION)
    inner_zone_mask = cv2.erode(phantom_mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)
    
    valid_edge_zone = cv2.bitwise_and(inner_zone_mask, cv2.bitwise_not(tool_mask))
    internal_edges = cv2.bitwise_and(edges, edges, mask=valid_edge_zone)

    close_k = odd_kernel(diagonal * CLOSE_FRACTION)
    connected_edges = cv2.morphologyEx(internal_edges, cv2.MORPH_CLOSE,
                                        np.ones((close_k, close_k), np.uint8))

    wound_contours, _ = cv2.findContours(connected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not wound_contours:
        return None, "no candidate contour after edge detection"

    candidates = sorted(wound_contours, key=cv2.contourArea, reverse=True)
    for c in candidates[:5]:
        valid, reason = validate_wound_contour(c, phantom_area)
        if valid:
            final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
            dilate_k = odd_kernel(diagonal * DILATE_FRACTION)
            final_mask = cv2.dilate(final_mask, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)
            return final_mask, "ok"

    _, reason = validate_wound_contour(candidates[0], phantom_area)
    return None, f"largest candidate rejected: {reason}"

# ==============================================================================
# BATCH DRIVER
# ==============================================================================
def process_batch():
    for run in RUNS:
        input_dir = os.path.join(BASE_DATASET_DIR, run, "images")
        output_dir = os.path.join(BASE_DATASET_DIR, run, "mask_openCV")

        image_files = []
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))

        if not image_files:
            print(f"No images found in {input_dir}. Skipping {run}.")
            continue

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[{run}] Found {len(image_files)} image(s). Saving masks to {output_dir}/")

        review_log = []
        saved_count = 0

        for image_path in image_files:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image = cv2.imread(image_path)
            if image is None:
                review_log.append(f"{base_name}: could not load image")
                continue

            mask, status = generate_mask(image)
            if mask is not None:
                output_path = os.path.join(output_dir, f"{base_name}_mask.jpg")
                cv2.imwrite(output_path, mask)
                saved_count += 1
            else:
                review_log.append(f"{base_name}: {status}")

        print(f"  {saved_count}/{len(image_files)} masks saved. "
              f"{len(review_log)} flagged for review.")

        if review_log:
            review_path = os.path.join(output_dir, "_REVIEW_NEEDED.txt")
            with open(review_path, "w") as f:
                f.write(f"{len(review_log)} of {len(image_files)} images did not produce "
                        f"a plausible wound mask and were skipped:\n\n")
                f.write("\n".join(review_log))
            print(f"  See {review_path} for the list.")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    process_batch()