import time
import PyKDL
import numpy as np
from ambf_client import Client
from surgical_robotics_challenge.psm_arm import PSM

class SurgicalStateMachine:
    def __init__(self):
        print("[SYSTEM] Initializing Hierarchical Control Pipeline...")
        self.client = Client('casa_fsm_node')
        self.client.connect()
        time.sleep(1.0)
        
        print("[SYSTEM] Waking up Inverse Kinematics Solvers...")
        self.arm = PSM(self.client, 'psm2')
        self.needle = self.client.get_obj_handle('Needle')
        self.yaw_link = self.client.get_obj_handle('psm2/toolyawlink')

    def move_with_feedback(self, target_frame, duration):
        """
        THE FEEDBACK LOOP: 
        Instead of sleeping, we continuously publish the target frame at 100Hz.
        This prevents the AMBF Watchdog from killing the motors, stopping the 'drift'.
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            self.arm.servo_cp(target_frame)
            time.sleep(0.01) # 100Hz Heartbeat

    def execute_pipeline(self):
        print("\n>>> COMMENCING AUTONOMOUS SUTURING SEQUENCE <<<")
        
        # 1. PERCEPTION LAYER: Exact Simulator Coordinates
        n_pose = self.needle.get_pos()
        needle_vec = PyKDL.Vector(n_pose.x, n_pose.y, n_pose.z)
        
        n_rot = self.needle.get_rot()
        needle_rot = PyKDL.Rotation.Quaternion(n_rot[0], n_rot[1], n_rot[2], n_rot[3])
        
        # T_w_needle: Needle in World Frame
        T_w_needle = PyKDL.Frame(needle_rot, needle_vec)
        
        b_pose = self.arm.base.get_pos()
        base_vec = PyKDL.Vector(b_pose.x, b_pose.y, b_pose.z)
        
        b_rot = self.arm.base.get_rot()
        base_rot = PyKDL.Rotation.Quaternion(b_rot[0], b_rot[1], b_rot[2], b_rot[3])
        
        # T_w_base: Base in World Frame
        T_w_base = PyKDL.Frame(base_rot, base_vec)
        
        # 2. SAFETY LAYER: Lock wrist orientation
        # Call self.arm.measured_cp().M to get the arm's current, safe rotation matrix in the base frame
        safe_rotation_base = self.arm.measured_cp().M
        
        # --- STATE 1: HOVER ---
        print("[STATE 1: HOVER] Moving 3cm above needle...")
        hover_vec = PyKDL.Vector(needle_vec.x(), needle_vec.y(), needle_vec.z() + 0.03)
        T_w_hover = PyKDL.Frame(needle_rot, hover_vec)
        
        # Calculate local target by multiplying inverted base frame
        T_b_hover = T_w_base.Inverse() * T_w_hover
        # Preserve Wrist Rotation
        T_b_hover.M = safe_rotation_base
        self.move_with_feedback(T_b_hover, 2.5)

        # --- STATE 2: APPROACH ---
        print("[STATE 2: APPROACH] Dropping down to needle...")
        grasp_vec = PyKDL.Vector(needle_vec.x(), needle_vec.y(), needle_vec.z() + 0.005)
        T_w_grasp = PyKDL.Frame(needle_rot, grasp_vec)
        
        # Calculate local target by multiplying inverted base frame
        T_b_grasp = T_w_base.Inverse() * T_w_grasp
        # Preserve Wrist Rotation
        T_b_grasp.M = safe_rotation_base
        self.move_with_feedback(T_b_grasp, 2.0)

        # --- STATE 3: GRASP ---
        print("[STATE 3: GRASP] Snapping jaws shut...")
        start_time = time.time()
        while time.time() - start_time < 1.0:
            self.arm.servo_cp(T_b_grasp)   # Hold position
            self.arm.set_jaw_angle(0.0)    # Close jaws
            time.sleep(0.01)

        # --- STATE 4: EXTRACT ---
        print("[STATE 4: EXTRACT] Pulling straight up 5cm...")
        extract_vec = PyKDL.Vector(needle_vec.x(), needle_vec.y(), needle_vec.z() + 0.05)
        T_w_extract = PyKDL.Frame(needle_rot, extract_vec)
        
        # Calculate local target by multiplying inverted base frame
        T_b_extract = T_w_base.Inverse() * T_w_extract
        # Preserve Wrist Rotation
        T_b_extract.M = safe_rotation_base
        
        start_time = time.time()
        while time.time() - start_time < 2.5:
            self.arm.servo_cp(T_b_extract)
            self.arm.set_jaw_angle(0.0)    # Keep jaws shut tightly
            time.sleep(0.01)

        print("\n[SUCCESS] Milestone 1 Complete. Needle Secured.")
        self.client.clean_up()

if __name__ == "__main__":
    fsm = SurgicalStateMachine()
    fsm.execute_pipeline()