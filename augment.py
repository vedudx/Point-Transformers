import open3d as o3d
import numpy as np
import random

# Define the input and output file paths
input_file = 'input.xyzn'  # Replace with your input point cloud file (xyzn format)
output_file = 'output.xyzn'  # Replace with your desired output file path



parent_directory="giga_dataset_normal"

def get_max_xyzw_in_subfolder(parent_directory):
    max_values = {}

    for subfolder in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, subfolder)
        
        # Only process if it's a directory (subfolder)
        if os.path.isdir(subfolder_path):
            max_xyzw = 0
            for filename in os.listdir(subfolder_path):
                if filename.startswith(subfolder) and filename.endswith(".txt"):
                    # Extract xyzw value from filename
                    try:
                        xyzw = int(filename.replace(subfolder, '').replace('.txt', ''))
                        max_xyzw = max(max_xyzw, xyzw)
                    except ValueError:
                        continue
            max_values[subfolder] = max_xyzw

    return max_values
# Load the input point cloud
point_cloud = o3d.io.read_point_cloud(input_file)


# Function to apply random rotation to a point cloud
def apply_random_rotation(point_cloud):
    """Apply a random rotation to the point cloud."""
    angles = [random.uniform(-45, 45) for _ in range(3)]  # Random rotation angles in degrees
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(angles))
    point_cloud.rotate(R, center=(0, 0, 0))

# Function to rotate point cloud
def rotate_point_cloud(point_cloud, angle_degrees):
    """Rotate a point cloud around the Z-axis by the given angle in degrees."""
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(angle_degrees)))
    point_cloud.rotate(R, center=(0, 0, 0))

# Augment the dataset by rotating the point cloud and saving it
augmented_point_clouds = []
angles_to_rotate = [15, 30, 45]  # You can adjust the rotation angles as needed

for angle in angles_to_rotate:
    rotated_point_cloud = point_cloud.clone()
    rotate_point_cloud(rotated_point_cloud, angle)
    augmented_point_clouds.append(rotated_point_cloud)

# Combine the original and augmented point clouds
combined_point_clouds = [point_cloud] + augmented_point_clouds

# Save the combined point clouds to a new file
o3d.io.write_point_cloud(output_file, o3d.geometry.PointCloud.concatenate_points(combined_point_clouds))
