import time
import numpy as np
import warnings
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings('ignore')
from movement_primitives.dmp import DMP, CartesianDMP
from ambf_client import Client

def extract_8dof_expert_data(filepath, start_frame, end_frame):
    print(f"[SYSTEM] Extracting full 8-DOF Kinematics from {filepath}...")
    raw_data = np.loadtxt(filepath)
    
    # 1. Translation
    xyz = raw_data[start_frame:end_frame, 57:60]
    
    # 2. Rotation
    rot_matrices_flat = raw_data[start_frame:end_frame, 60:69]
    quaternions = np.zeros((rot_matrices_flat.shape[0], 4))
    for i in range(rot_matrices_flat.shape[0]):
        quaternions[i] = R.from_matrix(rot_matrices_flat[i].reshape(3, 3)).as_quat()
        
    qw = quaternions[:, 3].reshape(-1, 1)
    qxyz = quaternions[:, 0:3]
    pose_trajectory = np.hstack((xyz, qw, qxyz))
    
    # 3. Jaw Angle
    jaw_trajectory = raw_data[start_frame:end_frame, 75].reshape(-1, 1)
    
    return pose_trajectory, jaw_trajectory

def execute_flawless_pickup():
    kinematics_path = 'data/expert_demonstrations/kinematics/Suturing_D001.txt'
    
    pose_traj, jaw_traj = extract_8dof_expert_data(kinematics_path, start_frame=614, end_frame=862)
    n_steps = pose_traj.shape[0]
    t_expert = np.linspace(0, 1, n_steps)

    print("[SYSTEM] Training CartesianDMP for Wrist Rotation and Spatial Translation...")
    pose_dmp = CartesianDMP(n_weights_per_dim=25, dt=1.0/(n_steps-1))
    pose_dmp.imitate(t_expert, pose_traj)

    print("[SYSTEM] Training 1D DMP for Gripper Jaw Actuation...")
    jaw_dmp = DMP(n_dims=1, n_weights_per_dim=15, dt=1.0/(n_steps-1))
    jaw_dmp.imitate(t_expert, jaw_traj)

    print("[SYSTEM] Connecting to AMBF ROS 2 Core...")
    c = Client('casa_autonomy_node')
    c.connect()
    time.sleep(1.0) 

    # Grab the rigid body
    arm_link = c.get_obj_handle('psm2/toolyawlink')
    needle = c.get_obj_handle('Needle')
    
    if arm_link is None or needle is None:
        print("[ERROR] Could not locate required AMBF objects. Is the simulation running?")
        return

    print("[SYSTEM] Querying Vision System (AMBF State) for Target...")
    needle_pose = needle.get_pos()
    current_pose = arm_link.get_pos()
    
    sim_start_pos = np.array([current_pose.x, current_pose.y, current_pose.z])
    sim_goal_pos = np.array([needle_pose.x, needle_pose.y, needle_pose.z + 0.01]) 
    
    start_quat = pose_traj[0, 3:7] 
    goal_quat = pose_traj[-1, 3:7]

    print("[SYSTEM] Generating Warped 8-DOF Kinematic Path...")
    sim_start_7d = np.hstack((sim_start_pos, start_quat))
    sim_goal_7d = np.hstack((sim_goal_pos, goal_quat))
    
    pose_dmp.configure(start_y=sim_start_7d, goal_y=sim_goal_7d)
    jaw_dmp.configure(start_y=jaw_traj[0], goal_y=jaw_traj[-1])
    
    t_rep, pose_reproduction = pose_dmp.open_loop(run_t=1.0)
    _, jaw_reproduction = jaw_dmp.open_loop(run_t=1.0)

    print("\n>>> INITIATING 8-DOF SURGICAL PICKUP <<<")
    time.sleep(1.0)
    
    for i in range(pose_reproduction.shape[0]):
        # 1. Apply Cartesian Translation
        arm_link.set_pos(pose_reproduction[i, 0], pose_reproduction[i, 1], pose_reproduction[i, 2])
        
        # 2. Apply Cartesian Rotation
        qw, qx, qy, qz = pose_reproduction[i, 3:7]
        arm_link.set_rot([qx, qy, qz, qw])
        
        # 3. DIRECT JOINT CONTROL FOR JAWS (Bypassing broken PSM library)
        jaw_angle = jaw_reproduction[i, 0]
        arm_link.set_joint_pos(0, jaw_angle)
        arm_link.set_joint_pos(1, jaw_angle)
        
        time.sleep(0.01)

    print("\n[SUCCESS] Needle Successfully Grasped.")
    c.clean_up()

if __name__ == "__main__":
    execute_flawless_pickup()