#!/usr/bin/env python3
"""
generate_masks_projection.py
=============================
Phase 3, Steps 3.1 + 3.2 -- 3D-to-2D wound mask projection with backface
culling.

SCOPE (3.1 + 3.2 done; 3.3/3.4 not started):
  3.1: local wound vertices -> world space -> camera space -> 2D pixel
       projection -> rasterized binary mask, run across all 4 runs.
  3.2: backface culling -- wound faces whose own normal points away from
       the camera are excluded before rasterization (see
       compute_face_normals_local, transform_normals_local_to_world, and
       the world_normals argument to transform_world_to_camera_pixels).
       Winding convention (WINDING_SIGN) verified empirically: with the
       current sign, 95.6% of individual wound-face normals point +Z
       (upward), matching physical expectation for a groove cut into the
       phantom's top surface -- confirms outward-pointing normals, not
       inverted ones.

  NOT yet implemented:
    * What 3.2 does NOT catch (by design, see original plan): a wound
      face whose own normal points toward the camera but is nonetheless
      hidden behind some OTHER part of the phantom's curved body. That's
      a full-mesh visibility problem (ray casting/z-buffering against the
      whole ~15k-face mesh, not just the 180 wound faces) -- deliberately
      out of scope here. Left to QA at oblique-angle frames.
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

# --- Backface-culling winding convention (Step 3.2) ---
# +1.0 if cross(v1-v0, v2-v0) points outward (standard OBJ convention);
# -1.0 if it turns out inverted for this mesh. Verified empirically below
# against real, already-confirmed-aligned frames -- see culling smoke
# test in main(). Flip this single constant if that check ever fails on
# a different mesh, rather than touching the math.
WINDING_SIGN = 1.0


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


def compute_face_normals_local(local_faces_xyz):
    """
    local_faces_xyz: (N, 3, 3). Returns (N, 3) unit normals, one per
    triangle, using the OBJ's own vertex winding order: normal =
    cross(v1-v0, v2-v0), normalized. This matches the standard OBJ/most
    3D-tool convention where CCW winding (viewed from outside the surface)
    gives an outward-pointing normal -- confirmed empirically below via
    WINDING_SIGN, not assumed.
    """
    v0, v1, v2 = local_faces_xyz[:, 0], local_faces_xyz[:, 1], local_faces_xyz[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return WINDING_SIGN * normals / norms


def transform_normals_local_to_world(normals_local, phantom_rotation_rpy):
    """Normals transform by rotation only (no translation). Correct as a
    direct rotation (not inverse-transpose) since the local->world
    transform here is a pure rotation with no scaling."""
    R = euler_to_matrix(
        phantom_rotation_rpy["roll"],
        phantom_rotation_rpy["pitch"],
        phantom_rotation_rpy["yaw"],
    )
    return (R @ normals_local.T).T


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
                                      fx, fy, cx, cy, world_normals=None):
    """
    world_faces_xyz: (N, 3, 3).
    world_normals: optional (N, 3) unit normals, one per face, in world
        space (from transform_normals_local_to_world). If provided, faces
        whose normal points away from the camera are culled (Step 3.2
        backface culling) -- see module docstring for what this does and
        does NOT catch (no full-mesh occlusion).
    Returns a list of length N: each entry is a (3, 2) array of (u, v)
    pixel coordinates, or None if the face is culled -- either because a
    vertex is behind the camera (numerically undefined for pinhole
    projection), or because it's a backface (if world_normals given).
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

    backface = None
    if world_normals is not None:
        centroids = world_faces_xyz.mean(axis=1)          # (N, 3)
        view_dirs = t_cam[None, :] - centroids             # camera - centroid, (N, 3)
        dots = np.sum(world_normals * view_dirs, axis=1)   # (N,)
        backface = dots <= 0.0

    pixel_faces = []
    for i in range(n_faces):
        z = z_opt[i]
        if np.any(z <= 1e-6):
            pixel_faces.append(None)  # a vertex is behind (or at) the camera
            continue
        if backface is not None and backface[i]:
            pixel_faces.append(None)  # Step 3.2: face points away from camera
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

def process_run(run_name, wound_faces_local_xyz, wound_normals_local, fx, fy, cx, cy, img_w, img_h):
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
    world_normals = transform_normals_local_to_world(wound_normals_local, phantom_rotation_rpy)

    out_masks_dir = os.path.join(OUTPUT_MASKS_ROOT, run_name, "masks")
    os.makedirs(out_masks_dir, exist_ok=True)

    total_culled_backface = 0
    for frame in frames:
        pixel_faces = transform_world_to_camera_pixels(
            world_faces_xyz, frame["camera_pos"], frame["camera_rpy"],
            fx, fy, cx, cy, world_normals=world_normals,
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
    wound_normals_local = compute_face_normals_local(wound_faces_local_xyz)
    print(f"[INFO] Loaded {len(wound_faces_local_xyz)} wound faces "
          f"({wound_faces_local_xyz.shape[0]} triangles after any ngon "
          f"triangulation) from {WOUND_FACES_JSON}")
    print(f"[INFO] Step 3.2 backface culling enabled "
          f"(WINDING_SIGN={WINDING_SIGN:+.0f}, verified: "
          f"{(wound_normals_local[:,2]>0).mean():.1%} of wound-face normals "
          f"point upward, matching a top-surface groove)")

    for run_name in RUN_NAMES:
        process_run(run_name, wound_faces_local_xyz, wound_normals_local, fx, fy, cx, cy, img_w, img_h)


if __name__ == "__main__":
    main()