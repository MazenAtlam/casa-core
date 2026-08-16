#!/usr/bin/env python3
"""Quick check: project wound along world Z using the new look-at camera matrix."""
import json, math
import numpy as np

DATA = "/home/mazen-atlam/casa-core/casa_app/casa_perception/dataset/raw/run_01/camera_poses.json"
with open(DATA) as f:
    d = json.load(f)

pp = d["phantom_position"]
PHANTOM_POS = np.array([pp["x"], pp["y"], pp["z"]])
FX = FY = 461.04; CX, CY = 320.0, 240.0

def camera_lookat_matrix(cam_pos, target_pos):
    fwd = target_pos - cam_pos
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd); up /= np.linalg.norm(up)
    back = -fwd
    return np.column_stack([right, up, back])

def proj_angle(R, t_cam, wA, wB):
    def proj(pt):
        q = R.T @ (pt - t_cam)
        z = -q[2]
        if z <= 0: return None
        return (FX * q[0] / z + CX, FY * (-q[1]) / z + CY)
    pA, pB = proj(wA), proj(wB)
    if pA is None or pB is None: return None
    du, dv = pB[0]-pA[0], pB[1]-pA[1]
    ang = math.degrees(math.atan2(abs(du), abs(dv)))
    return ang, pA, pB

SPAN = 0.058
print(f"{'Wound axis':<15} {'f0(~0°)':>9} {'f1':>9} {'f2':>9} {'f3(~0°)':>9}")
print("-"*60)

for axis_name, axis_vec in [("world +Y", np.array([0,1,0])),
                              ("world +Z", np.array([0,0,1])),
                              ("world +X", np.array([1,0,0]))]:
    wA = PHANTOM_POS - SPAN * axis_vec
    wB = PHANTOM_POS + SPAN * axis_vec
    angles = []
    for fi in [0, 1, 2, 3]:
        fr = d["frames"][fi]
        cp = fr["camera_pos"]
        t_cam = np.array([cp["x"], cp["y"], cp["z"]])
        R = camera_lookat_matrix(t_cam, PHANTOM_POS)
        res = proj_angle(R, t_cam, wA, wB)
        if res is None: angles.append("behind")
        else: angles.append(f"{res[0]:5.1f}°")
    print(f"{axis_name:<15} {' '.join(f'{a:>9}' for a in angles)}")
