import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_expert_data():
    """Simulates an expert surgeon's suturing motion (a deep swoop)."""
    t = np.linspace(0, 2 * np.pi, 200)
    x = 0.05 * np.cos(t)           # 5cm lateral movement
    y = 0.02 * np.sin(2 * t)       # 2cm forward/back adjustment
    z = -0.08 * np.sin(t)          # 8cm depth plunge into tissue
    
    # Combine into a 200x3 matrix (Time x Spatial Coordinates)
    trajectory = np.vstack((x, y, z)).T
    np.save('expert_suturing_baseline.npy', trajectory)
    print("[SYSTEM] Synthetic expert dataset saved to 'expert_suturing_baseline.npy'")

def inspect_and_plot():
    """Loads the matrix and visualizes the spatial path."""
    print("[SYSTEM] Loading trajectory matrix...")
    data = np.load('expert_suturing_baseline.npy')
    
    print(f"[DATA] Matrix Shape: {data.shape}")
    print(f"[DATA] Data Points (Time steps): {data.shape[0]}")
    print(f"[DATA] Dimensions (X, Y, Z): {data.shape[1]}")
    
    # 3D Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the path
    ax.plot(data[:, 0], data[:, 1], data[:, 2], label='Expert Suturing Path', color='b', linewidth=2)
    
    # Mark Start and End points
    ax.scatter(data[0, 0], data[0, 1], data[0, 2], color='green', s=100, label='Start (Entry)')
    ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2], color='red', s=100, label='End (Exit)')
    
    ax.set_title("Expert Surgical Trajectory ($SE(3)$ Sub-space)")
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    ax.set_zlabel("Z (Depth in Meters)")
    ax.legend()
    
    print("[SYSTEM] Rendering 3D spatial plot...")
    plt.show()

if __name__ == "__main__":
    generate_synthetic_expert_data()
    inspect_and_plot()