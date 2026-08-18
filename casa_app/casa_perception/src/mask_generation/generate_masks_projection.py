#!/usr/bin/env python3
"""
generate_masks_projection.py
=============================
Phase 3, Step 3.1 -- 3D-to-2D wound mask projection.

SCOPE (Step 3.1 ONLY -- per explicit instruction, later steps not started):
  Implements the core geometry pipeline: local wound vertices (from the
  phantom OBJ) -> world space -> camera space -> 2D pixel projection ->
  rasterized binary mask, run across all 4 captured runs.

  NOT yet implemented (do not treat any of this as done):
    * Step 3.2 -- face-normal/backface occlusion culling of wound faces.
      This script only discards a face if any of its 3 vertices sits
      behind the camera plane (z_opt <= 0), which is a numerical
      requirement for the pinhole formula to be defined at all -- NOT
      the "does the phantom's own curved body block this face" check
      planned for 3.2.
    * Step 3.3 -- final images/ + masks/ paired output structure. This
      script writes masks only, under OUTPUT_MASKS_ROOT/<run>/masks/.
    * Step 3.4 -- QA pass (alignment, boundary tightness, etc).

CONFIRMED SCHEMA (from the real wound_faces_simple.json):
  {
    "obj_path": "<path to the source phantom OBJ, relative to wherever
                  Phase 1's extract script was run from>",
    "wound_face_indices": [...],    # indices into the OBJ's 'f' line list
                                     # -- NOT vertex-index triplets. Each
                                     # entry selects one whole OBJ face;
                                     # its vertex indices are then read
                                     # from that face's 'f' line.
    "wound_vertex_indices": [...],  # a separate flat list of vertex
                                     # indices (likely what Phase 1's "red
                                     # dots" preview used). NOT used here
                                     # -- face-based selection is what
                                     # actually reconstructs the wound
                                     # surface for rasterization.
    "curvature_percentile_used": ..., "merge_dist": ...
  }

  FACE_INDEX_BASE (below) assumes these are 0-based, matching typical
  Python mesh-tooling (e.g. trimesh) convention -- NOT verified against
  the real OBJ (not available in this session). A runtime bounds check
  raises immediately with a clear message if this assumption is wrong,
  rather than silently producing a garbage selection.

CAMERA INTRINSICS -- now derived from world_mono.yaml's `cameraL` block
  (the only active camera per `cameras: [cameraL]`, matching the
  stereo/left topic), via the standard pinhole relationship:
      fy = (img_h/2) / tan(fov_v/2);  fx = fy;  cx = img_w/2;  cy = img_h/2
  Two assumptions flagged inline where computed:
    - `field view angle` is treated as the VERTICAL fov (common engine
      convention, e.g. OpenGL's fovy) -- not independently confirmed from
      AMBF's source.
    - Square pixels (fx == fy) -- standard default, no evidence of
      non-square pixels in this config.
  With cameraL's actual values (55 deg fov, 640x480), this resolves to
  fx=fy~=461.04, cx=320, cy=240 -- computed at runtime, not hardcoded, so
  it stays correct if world_mono.yaml changes.

PHANTOM_OBJ_PATH -- set below to the path you gave (relative to wherever
  this script is run from). The script also cross-checks this against
  wound_faces.json's own embedded obj_path and warns on any mismatch.
  Note this path is relative, not absolute -- it will only resolve
  correctly if this script is run from the same working-directory depth
  Phase 1's script was run from. Worth converting to an absolute path
  once your project layout is fixed, to avoid this breaking silently if
  run from elsewhere later (e.g. a different terminal, a scheduled job).

ROTATION MATH -- confidence notes:
  euler_to_matrix() uses R = Rz(yaw) @ Ry(pitch) @ Rx(roll). For the
  CAMERA side, this is not just copied from orbit_scan.py's docstring
  comment -- it's independently derivable: look_at_rpy() computes pitch
  and yaw as the spherical (polar, azimuth) angles of a vector `a`, and
  R = Rz(yaw) @ Ry(pitch) is exactly the rotation that carries the local
  +Z axis onto a vector with those spherical angles. That forces local -Z
  (forward) to align with the true look direction in general, not just at
  rpy=(0,0,0) -- so the pitch/yaw part of this formula is high-confidence.
  Also, every camera_rpy in your captured data has roll == 0 exactly
  (look_at_rpy always returns 0.0 for roll), so the Rx(roll) term is
  inert (identity) for all real camera frames regardless of where it sits
  in the product -- that ambiguity doesn't affect your dataset.

  STILL UNVERIFIED (flagged, not blocking, check during Step 3.4):
    - The right/up axis pairing around that forward axis (this script
      assumes right = local +X, up = local +Y -- the common convention
      paired with forward = local -Z, but not empirically confirmed the
      way the forward axis was). If projected masks come out mirrored or
      rotated 90 degrees, flip RIGHT_SIGN / UP_SIGN below rather than
      touching the core math.
    - phantom_rotation_rpy may have nonzero roll (unlike camera_rpy, it's
      an arbitrary read, not derived from look_at_rpy). If so, the Rx-term
      placement in the product DOES matter for the phantom transform in a
      way it doesn't for the camera transform. Check your logged
      phantom_rotation_rpy roll value -- if it's near 0, this is moot; if
      it's substantially nonzero, this needs to be pinned down precisely
      before trusting projected masks.
"""

import os
import json
import math
import numpy as np
import cv2

# ==============================================================================
# CONFIG -- fill in / confirm before running (see docstring above)
# ==============================================================================
# ==============================================================================
# CONFIG -- all paths are anchored to this script's directory via __file__,
# so the script works correctly regardless of the working directory it is
# invoked from.
# ==============================================================================

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.dirname(_SCRIPT_DIR)           # .../src/
_PERCEPTION   = os.path.dirname(_SRC_DIR)               # .../casa_perception/
_SRG          = os.path.join(_SRC_DIR, "..", "..",
                              "surgical_robotics_challenge")  # .../casa_app/surgical_robotics_challenge/

RAW_RUNS_DIR      = os.path.join(_PERCEPTION, "dataset", "raw")
OUTPUT_MASKS_ROOT = os.path.join(_PERCEPTION, "dataset", "processed")

WOUND_FACES_JSON  = os.path.join(_SCRIPT_DIR, "..", "wound_extract",
                                   "output", "wound_faces_simple.json")

PHANTOM_OBJ_PATH  = os.path.join(_SRG, "ADF", "Phantoms", "Simple",
                                   "high_res", "Phantom.OBJ")

WORLD_YAML_PATH   = os.path.join(_SRG, "ADF", "world", "world_mono.yaml")
CAMERA_KEY        = "cameraL"   # only active camera per world_mono.yaml's `cameras:` list

RUN_NAMES = ["run_01", "run_02", "run_03", "run_04"]


# 0-based vs 1-based indexing for wound_face_indices -- see docstring.
FACE_INDEX_BASE = 0

# --- Camera axis-convention flags (see "STILL UNVERIFIED" in docstring) ---
RIGHT_SIGN = 1.0
UP_SIGN = 1.0


def load_camera_intrinsics(world_yaml_path, camera_key=CAMERA_KEY):
    """
    Derives (fx, fy, cx, cy, img_w, img_h) from the named camera block in
    world_mono.yaml, via the standard pinhole relationship. See docstring
    for the two flagged assumptions (fov is vertical; square pixels).
    """
    import yaml
    with open(world_yaml_path, "r") as f:
        world = yaml.safe_load(f)
    cam_cfg = world[camera_key]

    fov_v = cam_cfg["field view angle"]              # ASSUMED vertical fov, radians
    res = cam_cfg["publish image resolution"]
    img_w, img_h = res["width"], res["height"]

    fy = (img_h / 2.0) / math.tan(fov_v / 2.0)
    fx = fy                                            # ASSUMED square pixels
    cx = img_w / 2.0
    cy = img_h / 2.0
    return fx, fy, cx, cy, img_w, img_h


# ==============================================================================
# Geometry helpers
# ==============================================================================

def euler_to_matrix(roll, pitch, yaw):
    """
    R such that world_point = R @ local_point (+ translation elsewhere).
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- see docstring for why this is
    high-confidence for the camera side and what remains unverified for
    the phantom side.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def load_obj(obj_path):
    """
    Parses 'v x y z' and 'f ...' lines from a Wavefront OBJ.

    Returns:
      vertices: (V, 3) float64 array.
      faces: list, one entry per 'f' line IN FILE ORDER (this ordering is
             what wound_face_indices indexes into) -- each entry is a list
             of 0-based vertex indices for that face (3 for a triangle,
             4+ for an ngon, NOT yet triangulated -- see triangulate_face).

    Handles all standard face-line forms ('f v1 v2 v3', 'f v1/vt1 ...',
    'f v1/vt1/vn1 ...', 'f v1//vn1 ...') by using only the vertex-index
    component. OBJ indices are 1-based in the file; converted to 0-based
    here (before FACE_INDEX_BASE is applied to wound_face_indices, which
    is a separate, independent indexing question -- see docstring).
    """
    vertices = []
    faces = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]  # 1-based -> 0-based
                faces.append(idxs)
    return np.array(vertices, dtype=np.float64), faces


def triangulate_face(face_vertex_indices):
    """Fan-triangulates a face with >3 vertices. Returns a list of 3-tuples."""
    if len(face_vertex_indices) == 3:
        return [tuple(face_vertex_indices)]
    tris = []
    v0 = face_vertex_indices[0]
    for i in range(1, len(face_vertex_indices) - 1):
        tris.append((v0, face_vertex_indices[i], face_vertex_indices[i + 1]))
    return tris


def load_wound_local_vertices(wound_faces_path, obj_path_override=None):
    """
    Returns wound_faces_xyz: (N, 3, 3) -- N triangular wound faces (after
    fan-triangulating any ngons), each 3 vertices, each vertex 3 local-
    frame (phantom OBJ) coordinates. Uses wound_face_indices (whole-face
    selection), not wound_vertex_indices -- see docstring.
    """
    with open(wound_faces_path, "r") as f:
        wf = json.load(f)

    if "wound_face_indices" not in wf:
        raise KeyError(
            f"Expected a 'wound_face_indices' key in {wound_faces_path}, "
            f"found keys: {list(wf.keys())}."
        )
    wound_face_indices = wf["wound_face_indices"]

    obj_path = obj_path_override or wf.get("obj_path")
    if obj_path is None:
        raise ValueError("No OBJ path available -- set PHANTOM_OBJ_PATH or "
                          "ensure wound_faces.json has an 'obj_path' key.")
    embedded_path = wf.get("obj_path")
    if obj_path_override and embedded_path and obj_path_override != embedded_path:
        print(f"[WARN] PHANTOM_OBJ_PATH ({obj_path_override}) differs from "
              f"wound_faces.json's own obj_path ({embedded_path}). "
              f"Using PHANTOM_OBJ_PATH.")

    vertices, raw_faces = load_obj(obj_path)

    max_idx = max(wound_face_indices)
    min_idx = min(wound_face_indices)
    highest_needed = max_idx - FACE_INDEX_BASE
    if highest_needed < 0 or highest_needed >= len(raw_faces):
        raise IndexError(
            f"wound_face_indices range [{min_idx}, {max_idx}] doesn't fit "
            f"the OBJ's {len(raw_faces)} faces under FACE_INDEX_BASE="
            f"{FACE_INDEX_BASE}. If FACE_INDEX_BASE=0 fails, try 1 (or "
            f"vice versa) -- do not proceed with a value that raises here."
        )

    selected_triangles = []
    for face_idx in wound_face_indices:
        raw_face = raw_faces[face_idx - FACE_INDEX_BASE]
        selected_triangles.extend(triangulate_face(raw_face))

    wound_faces_xyz = np.array(
        [[vertices[i0], vertices[i1], vertices[i2]] for (i0, i1, i2) in selected_triangles],
        dtype=np.float64,
    )
    return wound_faces_xyz


def transform_local_to_world(local_faces_xyz, phantom_position, phantom_rotation_rpy):
    """local_faces_xyz: (N, 3, 3). world = R @ local + t, per vertex."""
    R = euler_to_matrix(
        phantom_rotation_rpy["roll"],
        phantom_rotation_rpy["pitch"],
        phantom_rotation_rpy["yaw"],
    )
    t = np.array([phantom_position["x"], phantom_position["y"], phantom_position["z"]])

    flat = local_faces_xyz.reshape(-1, 3)
    world_flat = (R @ flat.T).T + t
    return world_flat.reshape(local_faces_xyz.shape)


def transform_world_to_camera_pixels(world_faces_xyz, camera_pos, camera_rpy,
                                      fx, fy, cx, cy):
    """
    world_faces_xyz: (N, 3, 3).
    Returns a list of length N: each entry is a (3, 2) array of (u, v)
    pixel coordinates, or None if the face has any vertex behind the
    camera (numerically undefined for pinhole projection -- see docstring,
    this is NOT Step 3.2's occlusion culling).
    """
    R_cam = euler_to_matrix(camera_rpy["roll"], camera_rpy["pitch"], camera_rpy["yaw"])
    t_cam = np.array([camera_pos["x"], camera_pos["y"], camera_pos["z"]])

    n_faces = world_faces_xyz.shape[0]
    flat = world_faces_xyz.reshape(-1, 3)

    # world -> camera-local (right=+X, up=+Y, forward=-Z convention)
    cam_local = (R_cam.T @ (flat - t_cam).T).T

    # RESOLVED via --calibrate (known fixed camera position, known rpy, real
    # rendered frames -- zero ambiguity): x_opt=local_X, y_opt=-local_Y is
    # CORRECT. The earlier "swap" was wrong -- it was fit to frame_0000,
    # which turned out to be a bad reference frame (see orbit_scan.py fix:
    # frame_0000 was captured with zero settle time after its pose command,
    # letting image and pose metadata desync for that one frame only).
    # Confirmed consistent across: calib identity, calib yaw_p90, and
    # frame_0001, using real wound geometry and real camera poses.
    x_opt = (RIGHT_SIGN * cam_local[:, 0]).reshape(n_faces, 3)
    y_opt = (-UP_SIGN * cam_local[:, 1]).reshape(n_faces, 3)
    z_opt = (-cam_local[:, 2]).reshape(n_faces, 3)

    pixel_faces = []
    for i in range(n_faces):
        z = z_opt[i]
        if np.any(z <= 1e-6):
            pixel_faces.append(None)  # a vertex is behind (or at) the camera
            continue
        u = fx * (x_opt[i] / z) + cx
        v = fy * (y_opt[i] / z) + cy
        pixel_faces.append(np.stack([u, v], axis=1))  # (3, 2)

    return pixel_faces


def rasterize_mask(pixel_faces, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for face in pixel_faces:
        if face is None:
            continue
        pts = np.round(face).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


# ==============================================================================
# Main
# ==============================================================================

def process_run(run_name, wound_faces_local_xyz, fx, fy, cx, cy, img_w, img_h):
    run_dir = os.path.join(RAW_RUNS_DIR, run_name)
    poses_path = os.path.join(run_dir, "camera_poses.json")
    with open(poses_path, "r") as f:
        data = json.load(f)

    phantom_position = data["phantom_position"]
    phantom_rotation_rpy = data["phantom_rotation_rpy"]
    frames = data["frames"]

    world_faces_xyz = transform_local_to_world(
        wound_faces_local_xyz, phantom_position, phantom_rotation_rpy
    )

    out_masks_dir = os.path.join(OUTPUT_MASKS_ROOT, run_name, "masks")
    os.makedirs(out_masks_dir, exist_ok=True)

    for frame in frames:
        pixel_faces = transform_world_to_camera_pixels(
            world_faces_xyz, frame["camera_pos"], frame["camera_rpy"],
            fx, fy, cx, cy,
        )
        mask = rasterize_mask(pixel_faces, img_w, img_h)

        out_path = os.path.join(out_masks_dir, frame["image"])
        cv2.imwrite(out_path, mask)

    print(f"[{run_name}] wrote {len(frames)} masks -> {out_masks_dir}")


def main():
    fx, fy, cx, cy, img_w, img_h = load_camera_intrinsics(WORLD_YAML_PATH, CAMERA_KEY)
    print(f"[INFO] Camera intrinsics from {WORLD_YAML_PATH} ({CAMERA_KEY}): "
          f"fx=fy={fx:.2f}, cx={cx}, cy={cy}, {img_w}x{img_h}")

    wound_faces_local_xyz = load_wound_local_vertices(WOUND_FACES_JSON, PHANTOM_OBJ_PATH)
    print(f"[INFO] Loaded {len(wound_faces_local_xyz)} wound faces "
          f"({wound_faces_local_xyz.shape[0]} triangles after any ngon "
          f"triangulation) from {WOUND_FACES_JSON}")

    for run_name in RUN_NAMES:
        process_run(run_name, wound_faces_local_xyz, fx, fy, cx, cy, img_w, img_h)


if __name__ == "__main__":
    main()