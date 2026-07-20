import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress minor library warnings
warnings.filterwarnings('ignore')
from movement_primitives.dmp import DMP

def extract_gesture_trajectory(kinematics_file, transcription_file, target_gesture='G3'):
    """Parses JIGSAWS files to extract the 3D path of a specific surgical gesture."""
    print(f"[SYSTEM] Scanning transcriptions for {target_gesture}...")
    
    start_frame, end_frame = None, None
    
    # 1. Parse the transcription text file
    with open(transcription_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == target_gesture:
                # JIGSAWS frames are 1-indexed, Python arrays are 0-indexed
                start_frame = int(parts[0]) - 1
                end_frame = int(parts[1]) - 1
                print(f"[DATA] Found '{target_gesture}' between frames {start_frame} and {end_frame}")
                break # Grab the first instance of this gesture
                
    if start_frame is None:
        raise ValueError(f"Error: Gesture {target_gesture} not found in {transcription_file}")

    # 2. Load the full raw kinematics matrix
    print("[SYSTEM] Loading full spatial kinematics matrix...")
    raw_data = np.loadtxt(kinematics_file)
    
    # 3. Slice the data: Rows (Time) and Columns 57-59 (Slave Right XYZ)
    primitive_trajectory = raw_data[start_frame:end_frame, 57:60]
    
    return primitive_trajectory

def train_and_visualize_primitive():
    # --- IMPORTANT: UPDATE THESE PATHS TO YOUR EXPERT 'E' FILES LATER ---
    kinematics_path = 'data/expert_demonstrations/kinematics/Suturing_B001.txt'
    transcription_path = 'data/expert_demonstrations/transcriptions/Suturing_B002.txt'
    
    # Extract the "Pushing Needle" primitive (G3)
    try:
        expert_trajectory = extract_gesture_trajectory(kinematics_path, transcription_path, 'G3')
    except Exception as e:
        print(e)
        return

    n_steps = expert_trajectory.shape[0]
    n_dims = expert_trajectory.shape[1]
    t_expert = np.linspace(0, 1, n_steps)
    
    # Initialize the DMP for 3-DOF spatial movement
    print(f"[SYSTEM] Initializing DMP for Isolated Surgical Primitive...")
    dmp = DMP(n_dims=n_dims, n_weights_per_dim=25, dt=1.0/(n_steps-1))
    
    # Train the DMP on the isolated gesture
    print("[SYSTEM] Executing One-Shot Imitation Learning...")
    dmp.imitate(t_expert, expert_trajectory)
    print("[SUCCESS] Primitive G3 Successfully Encoded!")
    
    # Autonomous Reproduction
    start_pos = expert_trajectory[0]
    goal_pos = expert_trajectory[-1]
    
    dmp.configure(start_y=start_pos, goal_y=goal_pos)
    t_reproduce, reproduction = dmp.open_loop(run_t=1.0)
    
    # Visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(expert_trajectory[:, 0], expert_trajectory[:, 1], expert_trajectory[:, 2], 
            label='JIGSAWS Expert (G3: Push Needle)', color='blue', linestyle='--', linewidth=3, alpha=0.7)
    
    ax.plot(reproduction[:, 0], reproduction[:, 1], reproduction[:, 2], 
            label='DMP Autonomous Reproduction', color='red', linewidth=2)
    
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=100, label='Needle Entry')
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color='red', s=100, label='Needle Exit')
    
    ax.set_title("Isolated Surgical Primitive: G3 (Needle Push)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    train_and_visualize_primitive()