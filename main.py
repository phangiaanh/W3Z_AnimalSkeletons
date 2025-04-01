from smal_torch.smal_torch import SMAL
import pickle
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def explore_pickle(pickle_path):
    """
    Explore and print the contents of a pickle file.
    Args:
        pickle_path: Path to the pickle file
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        def print_structure(obj, level=0, name="root"):
            indent = "  " * level
            if isinstance(obj, (dict, OrderedDict)):
                print(f"{indent}{name}: dict with {len(obj)} keys")
                for key, value in obj.items():
                    print_structure(value, level + 1, str(key))
            elif isinstance(obj, (list, tuple)):
                print(f"{indent}{name}: {type(obj).__name__} with {len(obj)} items")
                if len(obj) > 0:
                    print_structure(obj[0], level + 1, "first_item")
            elif isinstance(obj, np.ndarray):
                print(f"{indent}{name}: numpy array with shape {obj.shape}, dtype {obj.dtype}")
            else:
                print(f"{indent}{name}: {type(obj).__name__}")
        
        print(f"\nExploring pickle file: {pickle_path}")
        print("=" * 50)
        print_structure(data)
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def visualize_3d_vertices(vertices, title="3D Vertices"):
    """
    Visualize 3D vertices in a scatter plot with equal axis scales.
    Args:
        vertices: numpy array of shape (N, 3) containing x, y, z coordinates
        title: title for the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o', s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more interactive
    ax.view_init(elev=30, azim=45)
    
    # Get the limits for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    # Calculate the range for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # Find the greatest range value
    max_range = max(x_range, y_range, z_range)
    
    # Calculate the mid-points for each axis
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    
    # Set new limits based on the maximum range
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    plt.show()

def visualize_3d_mesh(vertices, faces, title="3D Mesh"):
    """
    Visualize 3D mesh using vertices and faces.
    Args:
        vertices: numpy array of shape (N, 3) containing x, y, z coordinates
        faces: numpy array of shape (M, 3) containing vertex indices for triangles
        title: title for the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh using triangles
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces,
                    color='lightgray',
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more interactive
    ax.view_init(elev=30, azim=45)
    
    # Get the limits for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    # Calculate the range for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # Find the greatest range value
    max_range = max(x_range, y_range, z_range)
    
    # Calculate the mid-points for each axis
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    
    # Set new limits based on the maximum range
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    pickle_paths = [
        "SMAL_horse/data/my_smpl_0000_horse_new_skeleton_horse.pkl",
        "SMAL_origin/data/smal_CVPR2017.pkl",
        "SMAL_origin/data/smal_CVPR2017_data.pkl",
        "SMAL_horse/data/walking_toy_symmetric_smal_0000_new_skeleton_pose_prior_new_36parts.pkl"
    ]
    
    for path in pickle_paths:
        explore_pickle(path)
        try:
            # Initialize SMAL model
            
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                # Try to find vertices and faces in the data
                if isinstance(data, dict):
                    vertices = None
                    faces = None
                    
                    # Check different possible keys for vertices
                    if 'v_template' in data:
                        vertices = data['v_template']
                    elif 'vertices' in data:
                        vertices = data['vertices']
                    
                    # Check different possible keys for faces
                    if 'faces' in data:
                        smal_model = SMAL(model_path=path, device='cpu')
                        faces = smal_model.faces.unsqueeze(0)
                        print(f"Faces {faces}")
                        faces = data['faces']
                        
                    elif 'f' in data:
                        smal_model = SMAL(model_path=path, device='cpu')
                        faces = smal_model.faces.unsqueeze(0)
                        print(f"Faces {faces}")
                        faces = data['f']
                    
                    if vertices is not None and faces is not None:
                        visualize_3d_mesh(vertices, faces, f"Mesh from {path}")
                        
        except Exception as e:
            print(f"Error visualizing mesh from {path}: {e}")
