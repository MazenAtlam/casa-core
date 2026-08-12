#!/usr/bin/env python3
"""
================================================================================
  extract_wound_from_mesh.py
  CASA -- Identify the Wound Region Directly From Mesh Geometry
================================================================================

WHY THIS SCRIPT EXISTS
------------------------
We confirmed (via a full repo search) that the wound has NO distinct color
or texture -- AMBF applies one flat uniform color to the whole phantom
(`use material: false` in the ADF). So the wound must be a genuine 3D
groove physically sculpted into the mesh, and that's exactly what this
finds: it looks for where the surface curves sharply (a groove bends
sharply; flat tissue doesn't), and returns exactly which mesh faces make
up the wound.

This only needs to run ONCE on the static mesh. After that, the
identified wound faces can be projected into ANY camera view (given that
frame's camera pose) to get a perfect, pixel-accurate ground-truth mask
for free -- no manual labeling per image required. That projection step
is separate and comes next; this script's job is just to answer "which
part of the mesh IS the wound."

VALIDATED
---------
Tested against the real Phantom.OBJ: the high-curvature vertices form a
clean, thin line down the exact center of the phantom (matching the
wound's visual location precisely), not scattered noise.

HOW IT WORKS
------------
1. Parse the OBJ (vertices + triangular faces).
2. For every vertex, look at all the faces touching it and measure how
   much their surface normals disagree (curvature proxy: 1 minus the
   length of the averaged unit normals -- flat regions average to ~1,
   sharply folded regions average to something much less than 1).
3. Keep only the most sharply curved vertices (top 5%).
4. Group them into connected clusters (using the mesh's own edge
   connectivity) -- the wound forms one large connected line; the
   block's sharp corners show up too, but as small, separate, isolated
   clusters, and get discarded by keeping only the largest cluster
   (plus any small fragments within a few mm of it, to bridge tiny gaps
   in the groove).
5. Convert the surviving wound VERTICES into wound FACES (any face
   where at least 2 of its 3 corners are wound vertices).
6. Save the result + a visualization to sanity-check by eye.

HOW TO RUN
----------
    pip install numpy scipy matplotlib --break-system-packages   # if not already present
    python3 extract_wound_from_mesh.py ../../../surgical_robotics_challenge/ADF/Phantoms/Simple/high_res/Phantom.OBJ --out-json wound_faces_simple.json --out-preview wound_mesh_preview_simple.png

    Outputs are always written to output/ next to this script, regardless of
    the working directory. The directory is created automatically if missing.

OUTPUT
------
    output/wound_faces.json       -- list of face indices (into the OBJ's
                                     face list) that make up the wound.
                                     This is the file the future mask-
                                     projection step needs.
    output/wound_mesh_preview.png -- top-down visualization for a manual
                                     sanity check before trusting the result.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CURVATURE_PERCENTILE = 95   # top X% most sharply curved vertices are candidates
MERGE_DIST = 0.005          # meters -- bridges small gaps in the groove; tune to
                             # your mesh's scale if results look fragmented
MIN_FACE_VOTES = 2          # a face counts as "wound" if >= this many of its
                             # 3 corners are wound vertices


# ==============================================================================
# OBJ PARSING
# ==============================================================================

def parse_obj(path):
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]  # OBJ is 1-indexed
                if len(idx) == 3:
                    faces.append(idx)
                else:
                    # fan-triangulate any non-triangular face
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts), np.array(faces)


# ==============================================================================
# CURVATURE
# ==============================================================================

def compute_face_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return normals / norms


def compute_vertex_curvature(verts, faces, face_normals):
    """Per-vertex curvature proxy: 1 - ||mean of adjacent face unit normals||.
    ~0 on flat surfaces, higher where the surface folds sharply."""
    n_verts = verts.shape[0]
    accum = np.zeros((n_verts, 3))
    counts = np.zeros(n_verts)
    for fi, (a, b, c) in enumerate(faces):
        for vi in (a, b, c):
            accum[vi] += face_normals[fi]
            counts[vi] += 1
    counts[counts == 0] = 1
    mean_normals = accum / counts[:, None]
    curvature = 1.0 - np.linalg.norm(mean_normals, axis=1)
    return curvature


def build_vertex_adjacency(faces):
    adj = defaultdict(set)
    for a, b, c in faces:
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    return adj


def largest_connected_component(candidate_verts, adj):
    visited = set()
    components = []
    for v in candidate_verts:
        if v in visited:
            continue
        stack, comp = [v], []
        visited.add(v)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb in candidate_verts and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return components


def merge_nearby_fragments(verts, components, merge_dist):
    """Keeps the largest component, and folds in any smaller fragment that's
    within merge_dist of it -- bridges small gaps without pulling in
    unrelated high-curvature spots elsewhere (like the block's corners)."""
    main = set(components[0])
    main_pts = verts[list(main)]
    tree = cKDTree(main_pts)

    for comp in components[1:]:
        pts = verts[comp]
        dists, _ = tree.query(pts, k=1)
        if dists.min() < merge_dist:
            main.update(comp)
    return main


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("obj_path", help="Path to the phantom's .OBJ file")
    parser.add_argument("--out-json", default="wound_faces.json",
                        help="Output JSON filename (saved inside output/ next to this script).")
    parser.add_argument("--out-preview", default="wound_mesh_preview.png",
                        help="Output preview PNG filename (saved inside output/ next to this script).")
    args = parser.parse_args()

    # Always resolve outputs relative to the script's own output/ directory,
    # regardless of the caller's working directory.
    out_json    = os.path.join(OUTPUT_DIR, os.path.basename(args.out_json))
    out_preview = os.path.join(OUTPUT_DIR, os.path.basename(args.out_preview))

    print(f"[1/5] Parsing {args.obj_path} ...")
    verts, faces = parse_obj(args.obj_path)
    print(f"      {len(verts)} vertices, {len(faces)} faces")

    print("[2/5] Computing per-vertex curvature ...")
    face_normals = compute_face_normals(verts, faces)
    curvature = compute_vertex_curvature(verts, faces, face_normals)

    thresh = np.percentile(curvature, CURVATURE_PERCENTILE)
    candidate_verts = set(np.where(curvature > thresh)[0].tolist())
    print(f"      {len(candidate_verts)} candidate high-curvature vertices "
          f"(top {100 - CURVATURE_PERCENTILE}%)")

    print("[3/5] Clustering and isolating the wound ...")
    adj = build_vertex_adjacency(faces)
    components = largest_connected_component(candidate_verts, adj)
    print(f"      {len(components)} separate clusters found "
          f"(largest={len(components[0])}, likely the wound; "
          f"small ones are probably the block's corners)")

    wound_verts = merge_nearby_fragments(verts, components, MERGE_DIST)
    print(f"      Final wound vertex count: {len(wound_verts)}")

    print("[4/5] Converting wound vertices to wound faces ...")
    wound_faces = []
    for fi, (a, b, c) in enumerate(faces):
        votes = sum(v in wound_verts for v in (a, b, c))
        if votes >= MIN_FACE_VOTES:
            wound_faces.append(fi)
    print(f"      {len(wound_faces)} wound faces identified")

    with open(out_json, "w") as f:
        json.dump({
            "obj_path": args.obj_path,
            "wound_face_indices": [int(i) for i in wound_faces],
            "wound_vertex_indices": [int(v) for v in sorted(wound_verts)],
            "curvature_percentile_used": CURVATURE_PERCENTILE,
            "merge_dist": MERGE_DIST,
        }, f, indent=2)
    print(f"[5/5] Saved: {out_json}")

    # visualization for a manual sanity check
    mask = np.zeros(len(verts), dtype=bool)
    mask[list(wound_verts)] = True
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.scatter(verts[~mask, 0], verts[~mask, 1], c="lightgray", s=2)
    ax.scatter(verts[mask, 0], verts[mask, 1], c="red", s=8)
    ax.set_aspect("equal")
    ax.set_title(f"Identified wound region ({mask.sum()} vertices, "
                 f"{len(wound_faces)} faces)")
    plt.tight_layout()
    plt.savefig(out_preview, dpi=120)
    print(f"      Saved preview: {out_preview}")
    print("\nLook at that preview image before trusting this -- it should be a")
    print("clean line/shape matching the wound, not scattered noise or the")
    print("block's outline. If it looks wrong, CURVATURE_PERCENTILE or")
    print("MERGE_DIST likely need adjusting at the top of this script.")


if __name__ == "__main__":
    main()