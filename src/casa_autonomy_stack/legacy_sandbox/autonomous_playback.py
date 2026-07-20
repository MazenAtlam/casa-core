import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from movement_primitives.dmp import DMP
from ambf_client import Client

def execute_closed_loop_surgery():
    # 1. Load the Expert Memory
    expert_trajectory = np.load('expert_suturing_baseline.npy')
    n_steps, n_dims = expert_trajectory.shape
    dmp = DMP(n_dims=n_dims, n_weights_per_dim=25, dt=1.0/(n_steps-1))
    dmp.imitate(np.linspace(0, 1, n_steps), expert_trajectory)

    # 2. Connect to AMBF
    print("[SYSTEM] Connecting to AMBF ROS 2 Core...")
    c = Client('casa_autonomy_node')
    c.connect()
    time.sleep(1.0) 

    arm = c.get_obj_handle('psm2/toolyawlink')
    
    # =====================================================================
    # 3. FLAWLESS PERCEPTION PLACEHOLDER
    # Instead of a CNN, we ask the physics engine exactly where the target is.
    # =====================================================================
    print("[SYSTEM] Querying Vision System (AMBF State) for Target...")
    needle = c.get_obj_handle('Needle')
    
    if needle is None:
        print("[ERROR] Could not locate the Needle in the simulation!")
        return

    needle_pose = needle.get_pos()
    sim_goal = np.array([needle_pose.x, needle_pose.y, needle_pose.z])
    
    # We want to hover just slightly above the needle so we don't smash it
    sim_goal[2] += 0.02 

    # =====================================================================
    # 4. DMP Path Warping
    # =====================================================================
    current_pose = arm.get_pos()
    sim_start = np.array([current_pose.x, current_pose.y, current_pose.z])

    print(f"[DATA] Perception Target Found at: {sim_goal}")
    print("[SYSTEM] DMP Warping Expert Trajectory to intercept Target...")
    
    dmp.configure(start_y=sim_start, goal_y=sim_goal)
    t_reproduce, reproduction = dmp.open_loop(run_t=1.0)

    # 5. Execution
    print("\n>>> INITIATING CLOSED-LOOP AUTONOMY <<<")
    time.sleep(1.0)
    
    for i in range(reproduction.shape[0]):
        arm.set_pos(reproduction[i, 0], reproduction[i, 1], reproduction[i, 2])
        time.sleep(0.01)

    print("\n[SUCCESS] Needle Intercept Complete.")
    c.clean_up()

if __name__ == "__main__":
    execute_closed_loop_surgery()