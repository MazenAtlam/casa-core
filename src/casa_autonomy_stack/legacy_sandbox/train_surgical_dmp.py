import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress minor warnings from the library
warnings.filterwarnings('ignore')

from movement_primitives.dmp import DMP

def train_and_evaluate_dmp():
    # 1. Load the Expert Trajectory
    print("[SYSTEM] Loading Expert Surgical Trajectory...")
    expert_trajectory = np.load('expert_suturing_baseline.npy')
    n_steps = expert_trajectory.shape[0]
    n_dims = expert_trajectory.shape[1]
    
    # Generate a normalized time array (0.0 to 1.0) for the expert data
    t_expert = np.linspace(0, 1, n_steps)

    # 2. Initialize the Dynamic Movement Primitive
    # We use 3 dimensions (X, Y, Z) and 30 Gaussian Basis Functions to capture the complex curve
    print(f"[SYSTEM] Initializing {n_dims}-DOF Dynamic Movement Primitive...")
    dmp = DMP(n_dims=n_dims, n_weights_per_dim=30, dt=1.0/(n_steps-1))

    # 3. Imitation Learning (Training)
    print("[SYSTEM] Executing Imitation Learning (Fitting Forcing Functions)...")
    dmp.imitate(t_expert, expert_trajectory)
    print("[SUCCESS] DMP Successfully Trained!")

    # 4. Autonomous Playback (Testing the Memory)
    print("[SYSTEM] Generating Autonomous Reproduction...")
    # We configure the DMP to start at the expert's start point and aim for the expert's end point
    start_pos = expert_trajectory[0]
    goal_pos = expert_trajectory[-1]
    
    dmp.configure(start_y=start_pos, goal_y=goal_pos)
    t_reproduce, reproduction = dmp.open_loop(run_t=1.0)

    # 5. Visualization: Compare Expert vs. AI Reproduction
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Expert
    ax.plot(expert_trajectory[:, 0], expert_trajectory[:, 1], expert_trajectory[:, 2], 
            label='Expert Demonstration', color='blue', linestyle='--', linewidth=3, alpha=0.6)
    
    # Plot AI Reproduction
    ax.plot(reproduction[:, 0], reproduction[:, 1], reproduction[:, 2], 
            label='DMP Autonomous Reproduction', color='red', linewidth=2)
    
    ax.set_title("Imitation Learning: Expert vs. DMP Reproduction")
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    ax.set_zlabel("Z (Depth)")
    ax.legend()
    
    print("[SYSTEM] Rendering Validation Plot...")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate_dmp()