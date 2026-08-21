#!/usr/bin/env python3
"""
qa_report.py
=============
Phase 3, Step 3.4 -- automated QA pass over the full processed dataset.

WHAT THIS DOES
--------------
Per the original Step 3.4 plan: don't eyeball all 2000 frames. Instead:
  1. COMPLETENESS check (fully automated, no sampling): for every run,
     confirm every frame listed in frame_poses.json has both an image
     and a mask file with the matching name, and flag any orphan image/
     mask files not listed in frame_poses.json either. This covers the
     original plan's "frame-mask pairing" check across all 4 runs, not
     just a spot-check.
  2. STATISTICAL OUTLIER detection on mask pixel counts (IQR-based) --
     catches empty masks, masks far smaller or larger than typical, and
     masks whose centroid sits near the image border (possible partial-
     visibility / off-frame wound).
  3. OVEREXPOSURE detection on the raw images -- a hard-edged white
     blowout was found in ~25% of a small hand-checked sample during
     manual verification; this folds that into the automated pass
     instead of relying on it being spotted by chance again.
  4. Writes a CSV report (qa_review/qa_report.csv) covering every frame,
     plus rendered mask-on-image overlay thumbnails for every flagged
     frame AND a random baseline sample of non-flagged frames (per the
     original plan: flagged frames + ~50 random baseline, not the full
     2000) -- so the actual manual look (alignment, boundary tightness,
     occlusion correctness, foreshortening -- these need a human, not a
     threshold) has a small, manageable, representative set to check
     instead of the whole dataset.

WHAT THIS DOES NOT DO
----------------------
Does not itself judge alignment, boundary tightness, occlusion
correctness, or foreshortening -- those need actual visual judgment per
the original plan. This script's job is narrowing 2000 frames down to a
small, representative, high-signal set for that manual look.

USAGE
-----
    python3 qa_report.py
Run from the same location generate_masks_projection.py expects
(dataset/processed/<run>/{images,masks}/ as produced by Step 3.3).

OUTPUT
------
    qa_review/qa_report.csv           -- one row per frame, every run
    qa_review/overlays/flag_*.png     -- overlay for each flagged frame
    qa_review/overlays/baseline_*.png -- overlay for the random sample
"""

import os
import csv
import json
import random

import numpy as np
import cv2

# ==============================================================================
# CONFIG
# ==============================================================================

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.dirname(_SCRIPT_DIR)
_PERCEPTION   = os.path.dirname(_SRC_DIR)
DATASET_ROOT  = os.path.join(_PERCEPTION, "dataset", "processed")

RUN_NAMES = ["run_01", "run_02", "run_03", "run_04"]
REVIEW_DIR = "qa_review"

RANDOM_BASELINE_N = 50     # per the original Step 3.4 plan
BORDER_MARGIN_FRAC = 0.05  # mask centroid within 5% of any edge -> flagged

# Whole-frame blowout: informational only, NOT a flag trigger
FRAME_BLOWOUT_INFO_THRESH = 0.10

# This is the check that actually matters: is the blowout anywhere near
# the wound itself, not just somewhere in the (often large, irrelevant)
# background.
WOUND_AREA_BLOWOUT_THRESH = 0.15
WOUND_AREA_DILATE_PX = 40  # margin around the mask to check, in pixels

RANDOM_SEED = 42           # reproducible baseline sample across reruns


# ==============================================================================
# Per-run loading and completeness check
# ==============================================================================

def list_run_frames(run):
    images_dir = os.path.join(DATASET_ROOT, run, "images")
    masks_dir = os.path.join(DATASET_ROOT, run, "masks")
    poses_path = os.path.join(images_dir, "frame_poses.json")
    with open(poses_path, "r") as f:
        data = json.load(f)
    frame_names = [fr["image"] for fr in data["frames"]]
    return images_dir, masks_dir, frame_names


def check_completeness(run, images_dir, masks_dir, frame_names):
    """Full (not sampled) frame/mask/frame_poses.json pairing check."""
    issues = []
    image_files = {fn for fn in os.listdir(images_dir) if fn.lower().endswith(".png")}
    mask_files = {fn for fn in os.listdir(masks_dir) if fn.lower().endswith(".png")}
    listed = set(frame_names)

    for name in sorted(listed - image_files):
        issues.append((run, name, "MISSING_IMAGE",
                       "listed in frame_poses.json but no image file found"))
    for name in sorted(listed - mask_files):
        issues.append((run, name, "MISSING_MASK",
                        "listed in frame_poses.json but no mask file found"))
    for name in sorted(image_files - listed):
        issues.append((run, name, "ORPHAN_IMAGE",
                        "image file exists but not listed in frame_poses.json"))
    for name in sorted(mask_files - listed):
        issues.append((run, name, "ORPHAN_MASK",
                        "mask file exists but not listed in frame_poses.json"))
    return issues


# ==============================================================================
# Per-frame metrics
# ==============================================================================

def mask_stats(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = mask.shape
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return {"n_px": 0, "cx": None, "cy": None, "w": w, "h": h}
    return {"n_px": len(xs), "cx": float(xs.mean()), "cy": float(ys.mean()), "w": w, "h": h}


def white_blowout_frac(image_path):
    """Whole-frame blowout fraction -- coarse signal, catches the artifact but
    doesn't distinguish 'background is white' from 'the wound itself is
    washed out'. Kept for visibility into how common the background artifact
    is, but see wound_area_blowout_frac() for the actually decision-relevant
    check."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return float((img > 250).all(axis=2).mean())


def wound_area_blowout_frac(image_path, mask_path, dilate_px=40):
    """Blowout fraction restricted to a margin around the wound itself
    (mask dilated by dilate_px), not the whole frame. This is the check
    that actually matters for training: a blown-out background elsewhere
    in frame doesn't compromise the segmentation target the way a blown-
    out wound region would. Returns None if the mask is empty (nothing to
    check a margin around)."""
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    if not np.any(mask > 127):
        return None
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    region = cv2.dilate((mask > 127).astype(np.uint8), kernel)
    blown = (img > 250).all(axis=2)
    region_bool = region > 0
    if region_bool.sum() == 0:
        return None
    return float(blown[region_bool].mean())


# ==============================================================================
# Main
# ==============================================================================

def main():
    random.seed(RANDOM_SEED)

    all_issues = []
    all_records = []

    for run in RUN_NAMES:
        images_dir, masks_dir, frame_names = list_run_frames(run)
        all_issues.extend(check_completeness(run, images_dir, masks_dir, frame_names))

        for name in frame_names:
            ipath = os.path.join(images_dir, name)
            mpath = os.path.join(masks_dir, name)
            if not (os.path.exists(ipath) and os.path.exists(mpath)):
                continue  # already captured as MISSING_IMAGE/MISSING_MASK above
            ms = mask_stats(mpath)
            if ms is None:
                continue
            frame_blow = white_blowout_frac(ipath)
            wound_blow = wound_area_blowout_frac(ipath, mpath, dilate_px=WOUND_AREA_DILATE_PX)
            all_records.append({
                "run": run, "name": name,
                "n_px": ms["n_px"], "cx": ms["cx"], "cy": ms["cy"],
                "w": ms["w"], "h": ms["h"],
                "frame_blowout": frame_blow, "wound_blowout": wound_blow,
            })

    if not all_records:
        print("[ERROR] No frames found -- check DATASET_ROOT / RUN_NAMES at the "
              "top of this script match your actual directory layout.")
        return

    # IQR-based outlier bounds on nonzero pixel counts (zero-pixel masks are
    # their own, separate EMPTY_MASK flag, not folded into this range)
    nonzero_counts = np.array([r["n_px"] for r in all_records if r["n_px"] > 0])
    q1, q3 = np.percentile(nonzero_counts, [25, 75])
    iqr = q3 - q1
    lo_bound, hi_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"[STATS] nonzero mask pixel counts: n={len(nonzero_counts)}, "
          f"median={np.median(nonzero_counts):.0f}, Q1={q1:.0f}, Q3={q3:.0f}, "
          f"IQR outlier bounds=[{max(0,lo_bound):.0f}, {hi_bound:.0f}]")

    flagged = []  # (run, name, [reasons])
    for r in all_records:
        reasons = []
        if r["n_px"] == 0:
            reasons.append("EMPTY_MASK")
        elif r["n_px"] < lo_bound:
            reasons.append(f"LOW_PIXEL_COUNT({r['n_px']})")
        elif r["n_px"] > hi_bound:
            reasons.append(f"HIGH_PIXEL_COUNT({r['n_px']})")

        if r["cx"] is not None:
            mx, my = BORDER_MARGIN_FRAC * r["w"], BORDER_MARGIN_FRAC * r["h"]
            if r["cx"] < mx or r["cx"] > r["w"] - mx or r["cy"] < my or r["cy"] > r["h"] - my:
                reasons.append("CENTROID_NEAR_BORDER")

        if r["wound_blowout"] is not None and r["wound_blowout"] > WOUND_AREA_BLOWOUT_THRESH:
            reasons.append(f"WOUND_AREA_OVEREXPOSED({r['wound_blowout']:.0%})")

        if reasons:
            flagged.append((r["run"], r["name"], reasons))

    print(f"[FLAGGED] {len(flagged)}/{len(all_records)} frames flagged for manual review")

    frame_blow_vals = [r["frame_blowout"] for r in all_records if r["frame_blowout"] is not None]
    n_bg_artifact = sum(1 for v in frame_blow_vals if v > FRAME_BLOWOUT_INFO_THRESH)
    print(f"[INFO] Background blowout artifact (>{FRAME_BLOWOUT_INFO_THRESH:.0%} of whole "
          f"frame, likely a finite backdrop-plane edge -- see prior verification): present "
          f"in {n_bg_artifact}/{len(frame_blow_vals)} frames ({n_bg_artifact/len(frame_blow_vals):.1%}). "
          f"NOT flagged on its own -- confirmed by hand across the full severity range that "
          f"this spares the wound region. Only flagged if it actually reaches the wound "
          f"(see WOUND_AREA_OVEREXPOSED below).")

    flagged_keys = {(run, name) for run, name, _ in flagged}
    candidates = [r for r in all_records if (r["run"], r["name"]) not in flagged_keys]
    baseline = random.sample(candidates, min(RANDOM_BASELINE_N, len(candidates)))
    baseline_keys = {(r["run"], r["name"]) for r in baseline}

    os.makedirs(REVIEW_DIR, exist_ok=True)
    report_path = os.path.join(REVIEW_DIR, "qa_report.csv")
    reasons_by_key = {(run, name): reasons for run, name, reasons in flagged}
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "name", "n_px", "cx", "cy", "frame_blowout_frac",
                          "wound_area_blowout_frac", "flag_reasons", "category"])
        for r in all_records:
            key = (r["run"], r["name"])
            reasons = reasons_by_key.get(key, [])
            category = "FLAGGED" if reasons else ("BASELINE" if key in baseline_keys else "")
            writer.writerow([r["run"], r["name"], r["n_px"], r["cx"], r["cy"],
                              r["frame_blowout"], r["wound_blowout"], ";".join(reasons), category])
    print(f"[REPORT] Wrote {report_path} ({len(all_records)} rows)")

    if all_issues:
        print(f"\n[COMPLETENESS] {len(all_issues)} filename-pairing issues found "
              f"(this is a FULL check across all frames, not a sample):")
        for run, name, kind, msg in all_issues:
            print(f"  {run}/{name}: {kind} -- {msg}")
    else:
        print(f"\n[COMPLETENESS] All {sum(len(list_run_frames(r)[2]) for r in RUN_NAMES)} "
              f"listed frames have a matching image + mask file. No pairing issues.")

    overlays_dir = os.path.join(REVIEW_DIR, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    to_render = [(run, name, "flag") for run, name, _ in flagged] + \
                [(r["run"], r["name"], "baseline") for r in baseline]
    n_written = 0
    for run, name, tag in to_render:
        ipath = os.path.join(DATASET_ROOT, run, "images", name)
        mpath = os.path.join(DATASET_ROOT, run, "masks", name)
        img = cv2.imread(ipath)
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        overlay = img.copy()
        overlay[mask > 127] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
        cv2.imwrite(os.path.join(overlays_dir, f"{tag}_{run}_{name}"), blended)
        n_written += 1

    print(f"\n[REVIEW] Wrote {n_written} overlay thumbnails to {overlays_dir}/ "
          f"({len(flagged)} flagged + {len(baseline)} random baseline)")
    print("[REVIEW] This is the set to actually look at -- alignment, boundary "
          "tightness, occlusion correctness, and foreshortening still need a "
          "human judgment call per frame, this script only narrowed down which "
          "frames are worth that look.")


if __name__ == "__main__":
    main()