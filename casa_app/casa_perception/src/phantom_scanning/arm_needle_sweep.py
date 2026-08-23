#!/usr/bin/env python3
"""
================================================================================
  arm_needle_sweep.py -- Move a grasped needle through frame near the wound
================================================================================

PURPOSE
-------
Runs ALONGSIDE orbit_scan.py (in a separate terminal, same running
simulator) so the captured dataset includes a tool/needle occluding the
wound sometimes -- teaching the segmentation model that a tool covering
part of the wound doesn't mean the wound isn't there.

Does NOT attempt real suturing accuracy -- it grasps the needle via the
repo's canonical Scene.task_3_setup_init(psm) flow (which calls
NeedleInitialization.move_to + jaw-close, exactly as in the challenge's
own task3_init_test.py example), then sweeps the gripper through a small
set of hovering waypoints above the phantom, on repeat.

STAGED VERIFICATION -- run these in order, don't skip to --sweep
------------------------------------------------------------------
    python3 arm_needle_sweep.py --test-grasp
        Grasps the needle, holds position, prints status. Watch the sim:
        did the needle actually latch onto the gripper? Ctrl-C to stop
        (releases the needle first). (doesn't work now - don't panic)

    python3 arm_needle_sweep.py --test-sweep --live
        Grasp, then sweep through waypoints, with a preview window so you
        can watch it move and confirm the motion looks sane (no violent
        collisions, stays near the phantom, needle stays gripped).

    python3 arm_needle_sweep.py --sweep
        The real thing, no preview -- run this in a second terminal while
        orbit_scan.py --capture runs in the first.

REPO FILES USED (surgical_robotics_challenge repo)
--------------------------------------------------
  surgical_robotics_challenge/psm_arm.py          -- PSM class
  surgical_robotics_challenge/scene.py            -- Scene.task_3_setup_init()
  surgical_robotics_challenge/simulation_manager.py  -- SimulationManager
  surgical_robotics_challenge/utils/task3_init.py -- NeedleInitialization
      (called internally by Scene.task_3_setup_init; not called here directly)

CONFIDENCE NOTE
----------------
All arm-control and grasp calls go through PSM.move_cp / PSM.set_jaw_angle
and Scene.task_3_setup_init -- the exact public API the repo documents and
uses in its own examples. What cannot be verified without a live sim run:
whether the hover waypoints below are collision-free for this phantom's
exact size/position -- that is why --test-sweep --live is a mandatory
checkpoint before the real run.
"""

import sys
import time
import argparse
import threading

import numpy as np
from PyKDL import Frame, Rotation, Vector

# ── repo classes (not copied, used as-is) ────────────────────────────────────
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.simulation_manager import SimulationManager

try:
    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    _HAVE_ROS_PREVIEW = True
except ImportError:
    _HAVE_ROS_PREVIEW = False


# ==============================================================================
# CONFIG
# ==============================================================================

PSM_GRASPER = "psm2"   # RIGHT arm -- grasps and sweeps the needle
PSM_OTHER   = "psm1"   # LEFT arm  -- mirrors on the opposite side

CAMERA_TOPIC = "/ambf/env/stereo/left/ImageData"   # only used for --live

# How far above the phantom surface the tips hover (meters).
# Lowered from 0.05 -- was reading as "very far away" in the screenshots.
HOVER_HEIGHT_M = 0.035

# How far to the side of the phantom centre each arm sits (meters).
# Doubled from 0.03 -- was producing very little visible left/right
# separation. PSM2 is on the local +x side; PSM1 is on the local -x side.
SIDE_OFFSET_M = 0.06

# Waypoints for PSM2 (grasper, RIGHT side of phantom).
# Each entry is (dx, dy, dz) in the PHANTOM'S OWN LOCAL frame (not raw
# world axes -- see _build_world_frame, which now rotates these through
# the phantom's actual orientation before adding its position). This is
# the fix for the X-crossing pattern in the screenshots: if the phantom
# has any real-world rotation, adding local offsets directly to world
# axes points them in the wrong direction entirely, not just "too close."
# dx = +SIDE_OFFSET_M -> right of centre, in the phantom's own frame.
# dy sweeps along the wound axis. Range widened to +/-0.06 to better
# match the wound's actual measured length (~-0.052 to +0.065 from the
# earlier mesh analysis).
GRASPER_WAYPOINTS = [
    ( SIDE_OFFSET_M, -0.06, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M, -0.03, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M,  0.00, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M,  0.03, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M,  0.06, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M,  0.03, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M,  0.00, HOVER_HEIGHT_M),
    ( SIDE_OFFSET_M, -0.03, HOVER_HEIGHT_M),
]

# Waypoints for PSM1 (other, LEFT side of phantom), same local-frame idea.
OTHER_WAYPOINTS = [
    (-SIDE_OFFSET_M,  0.06, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M,  0.03, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M,  0.00, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M, -0.03, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M, -0.06, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M, -0.03, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M,  0.00, HOVER_HEIGHT_M),
    (-SIDE_OFFSET_M,  0.03, HOVER_HEIGHT_M),
]

SECONDS_PER_WAYPOINT_MOVE = 3.0

# Gripper orientation -- pointing roughly downward at the phantom.
# Rotation.RPY(pi, 0, pi/2) is the known-good downward orientation used
# in the repo's own interface_via_method_api.py example.
GRASPER_ORIENTATION_RPY = (np.pi, 0.0,  np.pi / 2.0)   # PSM2 (right)
OTHER_ORIENTATION_RPY   = (np.pi, 0.0, -np.pi / 2.0)   # PSM1 (left, mirrored)

JAW_OPEN_ANGLE   = 0.8   # radians -- same value used in task3_init_test.py
JAW_CLOSED_ANGLE = 0.0


# ==============================================================================
# OPTIONAL LIVE PREVIEW  (only if --live and rclpy/cv2 are available)
# ==============================================================================

if _HAVE_ROS_PREVIEW:
    class CamSub(Node):
        def __init__(self):
            super().__init__("arm_needle_sweep_preview")
            self.bridge = CvBridge()
            self.frame  = None
            self.create_subscription(Image, CAMERA_TOPIC, self._cb, 10)

        def _cb(self, msg):
            try:
                self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError:
                pass

        def get_frame(self):
            return None if self.frame is None else self.frame.copy()


# ==============================================================================
# GRASP
# Uses Scene.task_3_setup_init(psm) -- the canonical repo method which:
#   1. Moves PSM to home joint pose (servo_jp)
#   2. Opens the jaw
#   3. Teleports the needle to the tip via NeedleInitialization.move_to()
#   4. Closes the jaw 30 times to latch the grasp
#   5. Releases the NeedleInitialization control loop (needle.release())
# Source: scene.py lines 94-112; task3_init_test.py
# ==============================================================================

def grasp_needle(simulation_manager, psm):
    """Grasp the needle using the repo's Scene.task_3_setup_init(psm)."""
    print("[GRASP] Running Scene.task_3_setup_init -- teleporting needle + closing jaws...")
    scene = Scene(simulation_manager)
    scene.task_3_setup_init(psm)
    print("[GRASP] Done.  Check sim: is the needle attached to the gripper?")


def release_needle(psm):
    """Open the jaws to drop the needle."""
    print("[GRASP] Releasing needle (opening jaws)...")
    for _ in range(30):
        psm.set_jaw_angle(JAW_OPEN_ANGLE)
        time.sleep(0.01)


def handoff_needle(psm_from, psm_to, phantom_frame):
    """
    Moves psm_to's gripper near psm_from's current (needle-holding) tip
    position, closes psm_to's jaws to try to latch the needle via the
    same physics-based grasp sensing described in the repo's grasp
    mechanism, then releases psm_from only after giving that a moment to
    register.

    CONFIDENCE NOTE -- this is the least certain piece of this script.
    Unlike a pose command (which either runs or doesn't), a successful
    grasp here depends on real-time physics: the ghost sensor bodies on
    psm_to actually detecting the needle within their sensing range at
    the moment the jaws close. Positioning that closely enough by an
    open-loop Cartesian move (no feedback on whether it actually worked)
    may take some tuning of HANDOFF_APPROACH_OFFSET below. That's exactly
    why --test-handoff exists as an isolated checkpoint -- watch the sim
    and confirm the needle actually transfers before combining this with
    the full sweep.
    """
    print(f"[HANDOFF] Moving receiving arm near the holding arm's tip...")

    # Where is the holding arm's tip right now, in world frame?
    T_from_base = _to_pykdl_frame(psm_from.measured_cp())   # tip pose in psm_from's own base frame
    T_from_world = psm_from.get_T_b_w() * T_from_base

    # Approach from slightly to the side, so the two grippers don't try
    # to occupy the exact same point (which could cause a physics glitch).
    approach_offset = Vector(0.0, 0.0, 0.01)
    T_to_target_world = Frame(T_from_world.M, T_from_world.p + approach_offset)
    T_to_target_base = psm_to.get_T_w_b() * T_to_target_world

    print("[HANDOFF] Opening receiving arm's jaw and moving into position...")
    psm_to.set_jaw_angle(JAW_OPEN_ANGLE)
    psm_to.move_cp(T_to_target_base, execute_time=2.0)
    time.sleep(2.5)

    print("[HANDOFF] Closing receiving arm's jaw (attempting grasp)...")
    for _ in range(40):
        psm_to.set_jaw_angle(JAW_CLOSED_ANGLE)
        time.sleep(0.01)

    print("[HANDOFF] Pausing to let the grasp register before releasing the first arm...")
    time.sleep(1.0)

    release_needle(psm_from)
    print("[HANDOFF] Done. Check the sim: did the needle actually transfer, "
          "or is it now on the ground / still on the original gripper?")


# ==============================================================================
# SWEEP  -- Cartesian moves through hover waypoints above the phantom
# ==============================================================================

def _build_world_frame(phantom_frame, local_offset, orientation_rpy):
    """
    Places a waypoint at `local_offset` (dx, dy, dz) IN THE PHANTOM'S OWN
    FRAME, then converts to world coordinates using the phantom's actual
    position AND rotation.

    This is the fix for the X-crossing pattern seen in testing: the
    previous version added (dx, dy, dz) directly to world-frame position,
    which is only correct if the phantom happens to be unrotated in the
    world. If it has any real rotation, "local +x" (meant to be "right of
    the wound") could point in a completely different world direction --
    verified with a numpy round-trip test: a 90-degree phantom rotation
    turns a "local +x" offset into a "world +y" offset, not "world +x".
    `phantom_frame` (a PyKDL Frame combining position + rotation) handles
    this correctly via direct multiplication.
    """
    local_point = Vector(*local_offset)
    world_point = phantom_frame * local_point
    return Frame(Rotation.RPY(*orientation_rpy), world_point)


def _to_pykdl_frame(cp):
    """
    psm.measured_cp() goes through the kinematics solver (self._kd.compute_FK),
    a different code path from position/pose reads -- and in this installed
    environment it returns a numpy matrix (4x4 homogeneous transform), not a
    PyKDL.Frame. Confirmed by the crash trace (routed through numpy's matrix
    multiply internals) and independently corroborated by another script in
    this repo that already handles this exact ambiguity via hasattr(cp, 'M').
    This normalizes either case to a real PyKDL.Frame.
    """
    if hasattr(cp, 'M'):
        return cp  # already a PyKDL.Frame
    rot = Rotation(cp[0, 0], cp[0, 1], cp[0, 2],
                   cp[1, 0], cp[1, 1], cp[1, 2],
                   cp[2, 0], cp[2, 1], cp[2, 2])
    pos = Vector(cp[0, 3], cp[1, 3], cp[2, 3])
    return Frame(rot, pos)


def _get_phantom_frame(simulation_manager, phantom_handle):
    """
    Reads the phantom's full pose (position + rotation) as a PyKDL Frame.

    Uses get_pos() + get_rotation() -- confirmed exact method names from
    simulation_manager.py's actual source (get_rot() does NOT exist; that
    was an incorrect guess last time, silently falling back to "assume
    unrotated" the whole time). Both go through units_conversion, the
    same reliable PyKDL-based path used by get_T_b_w()/get_T_w_b() --
    NOT the kinematics-solver path that returns numpy matrices for
    measured_cp(), so no normalization needed here.
    """
    pos = phantom_handle.get_pos()
    p = Vector(pos.x(), pos.y(), pos.z())
    try:
        rot = phantom_handle.get_rotation()
        print(f"[SETUP] Phantom rotation read successfully: {rot}")
        return Frame(rot, p)
    except Exception as e:
        print(f"[WARN] Could not read phantom rotation ({e}) -- assuming "
              f"unrotated (identity). If waypoints still look wrong after "
              f"this fix, this fallback firing is the likely reason.")
        return Frame(Rotation(), p)


def _world_to_psm_base(psm, T_world):
    """
    Convert a target pose from world frame to PSM-base frame.
    PSM.servo_cp / PSM.move_cp expect the pose w.r.t. the PSM's own base
    frame, not the world frame.  Conversion:  T_t_b = T_w_b * T_world
    (get_T_w_b() is documented in psm_arm.py lines 174-176).
    """
    T_w_b = psm.get_T_w_b()
    return T_w_b * T_world


def _pre_position_arm(psm, waypoints, phantom_frame, orientation_rpy, label):
    """
    Move an arm to the first waypoint of its path before the sweep loop
    starts, so both arms are already near the phantom (not at home) the
    moment the loop begins.
    """
    print(f"[SWEEP] Pre-positioning {label} to first waypoint...")
    first = waypoints[0]
    T_world = _build_world_frame(phantom_frame, first, orientation_rpy)
    T_base  = _world_to_psm_base(psm, T_world)
    # Use a longer execute_time so the arm doesn't rush to the phantom
    psm.move_cp(T_base, execute_time=SECONDS_PER_WAYPOINT_MOVE * 2)
    time.sleep(SECONDS_PER_WAYPOINT_MOVE * 2 + 0.5)
    print(f"[SWEEP] {label} in position.")


def run_sweep(psm_grasper, psm_other, phantom_frame, show_preview, cam_node=None):
    """
    Sweep both arms above the phantom -- each arm on its own side:
      PSM2 (grasper): +x side of phantom, sweeps along y (wound axis)
      PSM1 (other)  : -x side of phantom, sweeps along y in opposite phase
    Both arms use separate waypoint lists so they never cross paths.
    """
    n = max(len(GRASPER_WAYPOINTS), len(OTHER_WAYPOINTS))
    print(f"\n[SWEEP] {n} waypoints per arm, {SECONDS_PER_WAYPOINT_MOVE}s per move, "
          f"looping.  Ctrl-C to stop.\n")
    print("[SWEEP] PSM2 (grasper) sweeps the RIGHT (+x) side of the phantom.")
    print("[SWEEP] PSM1 (other)   sweeps the LEFT  (-x) side of the phantom.\n")

    # Move both arms to their starting waypoints before the loop
    _pre_position_arm(psm_grasper, GRASPER_WAYPOINTS, phantom_frame,
                      GRASPER_ORIENTATION_RPY, "PSM2/grasper")
    _pre_position_arm(psm_other,   OTHER_WAYPOINTS,   phantom_frame,
                      OTHER_ORIENTATION_RPY,   "PSM1/other")

    idx = 0
    try:
        while True:
            g_offset = GRASPER_WAYPOINTS[idx % len(GRASPER_WAYPOINTS)]
            o_offset = OTHER_WAYPOINTS[  idx % len(OTHER_WAYPOINTS)]

            # ── PSM2 -- right side, sweeps +x of phantom ─────────────────────
            T_world_g = _build_world_frame(phantom_frame, g_offset,
                                           GRASPER_ORIENTATION_RPY)
            T_base_g  = _world_to_psm_base(psm_grasper, T_world_g)
            psm_grasper.move_cp(T_base_g, execute_time=SECONDS_PER_WAYPOINT_MOVE)

            # ── PSM1 -- left side, sweeps -x of phantom ──────────────────────
            T_world_o = _build_world_frame(phantom_frame, o_offset,
                                           OTHER_ORIENTATION_RPY)
            T_base_o  = _world_to_psm_base(psm_other, T_world_o)
            psm_other.move_cp(T_base_o, execute_time=SECONDS_PER_WAYPOINT_MOVE)

            g_xy = (g_offset[0], g_offset[1])
            o_xy = (o_offset[0], o_offset[1])
            print(f"[SWEEP] step {idx:3d}  PSM2(right) dy={g_offset[1]:+.3f}  "
                  f"PSM1(left) dy={o_offset[1]:+.3f}")

            # ── wait out the move, optionally showing the camera feed ────────
            t_end = time.time() + SECONDS_PER_WAYPOINT_MOVE
            while time.time() < t_end:
                if show_preview and cam_node is not None:
                    frame = cam_node.get_frame()
                    if frame is not None:
                        cv2.putText(frame,
                                    f"step {idx}  R-dy={g_offset[1]:+.2f}  "
                                    f"L-dy={o_offset[1]:+.2f}",
                                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        cv2.imshow("Arm Sweep Preview", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n[STOP] Q pressed.")
                            return
                time.sleep(0.05)

            idx += 1

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl-C received.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sweep a grasped needle above the phantom for dataset occlusion.")
    parser.add_argument("--test-grasp", action="store_true",
                        help="Grasp only, hold, release on Ctrl-C.  Verify visually.")
    parser.add_argument("--test-handoff", action="store_true",
                        help="Grasp with PSM2, then attempt handoff to PSM1. "
                             "Isolated test -- watch closely, this is the least "
                             "certain part (depends on real-time physics grasp "
                             "sensing, not just a pose command).")
    parser.add_argument("--test-sweep", action="store_true",
                        help="Grasp then sweep.  Combine with --live for a preview window.")
    parser.add_argument("--sweep", action="store_true",
                        help="Production run -- grasp then sweep, no preview.  "
                             "Run alongside orbit_scan.py --capture.")
    parser.add_argument("--live", action="store_true",
                        help="Show a camera preview window (requires rclpy + cv2).")
    args = parser.parse_args()

    if not (args.test_grasp or args.test_handoff or args.test_sweep or args.sweep):
        parser.print_help()
        sys.exit(1)

    if args.live and not _HAVE_ROS_PREVIEW:
        print("[WARN] --live requested but rclpy/cv2/cv_bridge not available. "
              "Continuing without preview.")

    # ── connect ───────────────────────────────────────────────────────────────
    print("[SETUP] Connecting via SimulationManager...")
    simulation_manager = SimulationManager('arm_needle_sweep')
    print("[SETUP] Waiting for AMBF state topics to populate (3 s)...")
    time.sleep(3.0)

    # ── PSM handles (detect_tool_id=False avoids a crash when the tool-id
    #    topic hasn't been published yet at startup) ───────────────────────────
    psm_g = PSM(simulation_manager, PSM_GRASPER, detect_tool_id=False)
    psm_o = PSM(simulation_manager, PSM_OTHER,   detect_tool_id=False)

    if not (psm_g.is_present() and psm_o.is_present()):
        print(f"[ERROR] Could not get handles for {PSM_GRASPER} and/or {PSM_OTHER}. "
              "Is the simulator running?")
        sys.exit(1)

    # ── phantom pose (position + rotation -- hover waypoints are placed
    #    relative to this, in the phantom's OWN frame, see _build_world_frame)
    phantom = simulation_manager.get_obj_handle("Phantom")
    time.sleep(0.5)
    phantom_frame = _get_phantom_frame(simulation_manager, phantom)
    print(f"[SETUP] Phantom world position: "
          f"({phantom_frame.p.x():.4f}, {phantom_frame.p.y():.4f}, {phantom_frame.p.z():.4f})")

    # ── optional live preview ─────────────────────────────────────────────────
    cam_node    = None
    spin_thread = None
    show_preview = args.live and _HAVE_ROS_PREVIEW
    if show_preview:
        if not rclpy.ok():
            rclpy.init()
        cam_node = CamSub()
        spin_thread = threading.Thread(
            target=rclpy.spin, args=(cam_node,), daemon=True)
        spin_thread.start()

    # ── run ───────────────────────────────────────────────────────────────────
    try:
        if args.test_grasp:
            grasp_needle(simulation_manager, psm_g)
            print("[TEST] Holding.  Press Ctrl-C to release and exit.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
            release_needle(psm_g)

        elif args.test_handoff:
            grasp_needle(simulation_manager, psm_g)
            print("[TEST] PSM2 holding needle. Attempting handoff to PSM1 in 2s...")
            time.sleep(2.0)
            handoff_needle(psm_g, psm_o, phantom_frame)
            print("[TEST] Holding final state. Press Ctrl-C to exit.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

        elif args.test_sweep or args.sweep:
            grasp_needle(simulation_manager, psm_g)
            run_sweep(psm_g, psm_o, phantom_frame, show_preview, cam_node)

    finally:
        if show_preview:
            cv2.destroyAllWindows()
            if cam_node is not None:
                cam_node.destroy_node()
            rclpy.shutdown()
            if spin_thread is not None:
                spin_thread.join(timeout=2.0)

    print("[DONE]")


if __name__ == "__main__":
    main()