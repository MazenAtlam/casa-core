import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from movement_primitives.dmp import DMP
from ambf_client import Client

def extract_4dof_expert_data(filepath, start_frame, end_frame):
    print(f"[SYSTEM] Extracting Translation and Jaw Kinematics from {filepath}...")
    raw_data = np.loadtxt(filepath)
    
    # 1. Extract ONLY 3D Translation (XYZ)
    xyz_trajectory = raw_data[start_frame:end_frame, 57:60]
    
    # 2. Extract Gripper Jaw Angle (1D)
    jaw_trajectory = raw_data[start_frame:end_frame, 75].reshape(-1, 1)
    
    return xyz_trajectory, jaw_trajectory

def execute_stable_pickup():
    kinematics_path = 'data/expert_demonstrations/kinematics/Suturing_D001.txt'
    
    xyz_traj, jaw_traj = extract_4dof_expert_data(kinematics_path, start_frame=614, end_frame=862)
    n_steps = xyz_traj.shape[0]
    t_expert = np.linspace(0, 1, n_steps)

    print("[SYSTEM] Training Spatial DMP for the Swoop...")
    spatial_dmp = DMP(n_dims=3, n_weights_per_dim=25, dt=1.0/(n_steps-1))
    spatial_dmp.imitate(t_expert, xyz_traj)

    print("[SYSTEM] Training 1D DMP for the Jaw Snap...")
    jaw_dmp = DMP(n_dims=1, n_weights_per_dim=15, dt=1.0/(n_steps-1))
    jaw_dmp.imitate(t_expert, jaw_traj)

    print("[SYSTEM] Connecting to AMBF ROS 2 Core...")
    c = Client('casa_autonomy_node')
    c.connect()
    time.sleep(1.0) 

    arm_link = c.get_obj_handle('psm2/toolyawlink')
    needle = c.get_obj_handle('Needle')
    
    if arm_link is None or needle is None:
        print("[ERROR] Could not locate required AMBF objects.")
        return

    print("[SYSTEM] Querying Vision System for Target...")
    needle_pose = needle.get_pos()
    current_pose = arm_link.get_pos()
    
    sim_start_pos = np.array([current_pose.x, current_pose.y, current_pose.z])
    
    # Target the needle, hovering slightly above it to prevent physics collisions
    sim_goal_pos = np.array([needle_pose.x, needle_pose.y, needle_pose.z + 0.012]) 

    print("[SYSTEM] Generating Autonomous Path...")
    spatial_dmp.configure(start_y=sim_start_pos, goal_y=sim_goal_pos)
    jaw_dmp.configure(start_y=jaw_traj[0], goal_y=jaw_traj[-1])
    
    t_rep, spatial_reproduction = spatial_dmp.open_loop(run_t=1.0)
    _, jaw_reproduction = jaw_dmp.open_loop(run_t=1.0)

    print("\n>>> INITIATING STABLE SURGICAL PICKUP <<<")
    time.sleep(1.0)
    
    for i in range(spatial_reproduction.shape[0]):
        # Apply the beautiful XYZ swoop
        arm_link.set_pos(spatial_reproduction[i, 0], spatial_reproduction[i, 1], spatial_reproduction[i, 2])
        
        # We explicitly DO NOT call set_rot() here. 
        # This prevents the "Baltimore Coordinate" physics explosion.
        
        # Actuate the jaws using native joint indices
        jaw_angle = jaw_reproduction[i, 0]
        arm_link.set_joint_pos(0, jaw_angle)
        arm_link.set_joint_pos(1, jaw_angle)
        
        # Slightly slower playback allows the physics engine to calculate joint drag smoothly
        time.sleep(0.015)

    print("\n[SUCCESS] Needle Successfully Grasped.")
    c.clean_up()

if __name__ == "__main__":
    execute_stable_pickup()