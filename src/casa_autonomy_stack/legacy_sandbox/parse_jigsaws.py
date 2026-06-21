import numpy as np
import matplotlib.pyplot as plt

def parse_expert_kinematics(filepath):
    print(f"[SYSTEM] Ingesting JIGSAWS dataset: {filepath}")
    
    # Load the space-separated text file into a NumPy matrix
    # This automatically handles the format inside Suturing_D001.txt
    raw_data = np.loadtxt(filepath)
    
    print(f"[DATA] Raw Kinematics Shape: {raw_data.shape} (Frames x 76 variables)")
    
    # According to readme.txt, Slave Right XYZ is columns 58-60 (0-indexed as 57, 58, 59)
    slave_right_xyz = raw_data[:, 57:60]
    
    print(f"[DATA] Extracted Slave Right Arm XYZ Shape: {slave_right_xyz.shape}")
    
    # Save the parsed, clean data for the DMP
    output_filename = 'expert_suturing_baseline.npy'
    np.save(output_filename, slave_right_xyz)
    print(f"[SYSTEM] Clean expert trajectory saved as {output_filename}")
    
    # Visualize the Expert's path
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(slave_right_xyz[:, 0], slave_right_xyz[:, 1], slave_right_xyz[:, 2], 
            label='JIGSAWS Expert Trajectory (Right Arm)', color='blue', linewidth=2)
    ax.scatter(slave_right_xyz[0, 0], slave_right_xyz[0, 1], slave_right_xyz[0, 2], 
               color='green', s=100, label='Start')
    
    ax.set_title("JIGSAWS Expert Surgical Path (Slave Right Arm)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Make sure to point this to an EXPERT (E) file, not a Novice (N) file!
    # Update the path below to wherever your kinematics folder is located
    parse_expert_kinematics('data/expert_demonstrations/kinematics/Suturing_D001.txt')