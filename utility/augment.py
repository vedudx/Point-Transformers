import open3d as o3d
import numpy as np
import os
import random
from pathlib import Path
import sys
parent_directory = sys.argv[1] #"giga_dataset_normal_cleaned/"

new_parent_directory = sys.argv[1]+"_augment/" #giga_small_dataset_normal_augment_cleaned/"


magicNo = 4000
random_rotation = True

print(f"Hyperparameters, magic no: {magicNo}, random rotation: {random_rotation}")

# Load the list of filenames to skip
with open("giga_dataset_normal_cleaned/modelnet5_test.txt", 'r') as file:
    skip_files = file.readlines()
skip_files = set(f.strip() for f in skip_files)


def clone_pointcloud(pcd):
    new_pcd = o3d.cuda.pybind.geometry.PointCloud()
    new_pcd.points = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_normals():
        new_pcd.normals = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(pcd.normals))
    if pcd.has_colors():
        new_pcd.colors = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(pcd.colors))
    return new_pcd

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
                        xyzw = int(filename.split('_')[-1].replace('.txt', ''))
                        max_xyzw = max(max_xyzw, xyzw)
                    except ValueError:
                        continue
            max_values[subfolder] = max_xyzw

    return max_values

# Function to apply random rotation to a point cloud
def apply_random_rotation(point_cloud):
    """Apply a random rotation to the point cloud."""
    angles = [random.uniform(-45, 45) for _ in range(3)]  # Random rotation angles in degrees
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(angles))
    point_cloud.rotate(R, center=(0, 0, 0))


def write_point_cloud(filename, point_cloud):
    with open(filename, 'w') as output_file:
    # Iterate through the points in the point cloud
        for i in range(len(point_cloud.points)):
            point = point_cloud.points[i]
            normal = point_cloud.normals[i]

            output_file.write(f"{point[0]},{point[1]},{point[2]},{normal[0]},{normal[1]},{normal[2]}\n")

def rotate_point_cloud(point_cloud, angle_degrees):
    """Rotate a point cloud around the Z-axis by the given angle in degrees."""
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(angle_degrees)))
    point_cloud.rotate(R, center=(0, 0, 0))

# Iterate over each subfolder
max_values = get_max_xyzw_in_subfolder(parent_directory)
angles_to_rotate = [15, 30, 45]  

sanity = 0
for subfolder, max_xyzw in max_values.items():
    subfolder_path = os.path.join(parent_directory, subfolder)

    if 'larch' in subfolder or 'snag' in subfolder:
        continue

    newSubFolder = Path(new_parent_directory + "/" + subfolder)
    newSubFolder.mkdir(parents=True, exist_ok=True)
    count = 1
    file_count = len(os.listdir(subfolder_path))
    print(f"no of files in {subfolder}: {file_count}")

    #copy all original files to the output

    for filename in os.listdir(subfolder_path):
        check_name = filename.replace(".txt", "")
        if filename.startswith(subfolder) and filename.endswith(".txt"):
            # Extract xyzw value from filename
            xyzw = int(filename.split('_')[-1].replace('.txt', ''))
            #xyzw = int(filename.replace(subfolder, '').replace('.txt', ''))
            input_file = os.path.join(subfolder_path, filename)
            new_input_file = new_parent_directory + "/" + subfolder + "/" + f"{check_name}.txt"
            point_cloud = o3d.io.read_point_cloud(input_file, format='xyzn', print_progress=True)

            write_point_cloud(new_input_file, point_cloud)


    for filename in os.listdir(subfolder_path):
        check_name = filename.replace(".txt", "")
    
    
        if filename.startswith(subfolder) and filename.endswith(".txt"):
            # Extract xyzw value from filename
            xyzw = int(filename.split('_')[-1].replace('.txt', ''))
            #xyzw = int(filename.replace(subfolder, '').replace('.txt', ''))
            
            input_file = os.path.join(subfolder_path, filename)

            new_input_file = new_parent_directory + "/" + subfolder + "/" + f"{check_name}.txt"
            point_cloud = o3d.io.read_point_cloud(input_file, format='xyzn', print_progress=True)

            write_point_cloud(new_input_file, point_cloud)

            if check_name in skip_files:
                print("gotcha")
                sanity += 1
                continue
            

            if count + file_count >= magicNo:
                break


            #Apply random rotations

            if random_rotation:
                rotated_point_cloud = clone_pointcloud(point_cloud)
                apply_random_rotation(rotated_point_cloud)
                new_xyzw = max_xyzw + count
                #print(filename, new_xyzw)

                #new_file_name
                output_file = new_parent_directory + "/" + subfolder + "/" + f"{subfolder}_{new_xyzw}.txt"
            # print(output_file)
                write_point_cloud(output_file, rotated_point_cloud)
                #output_file = os.path.join(subfolder_path, f"{subfolder}{new_xyzw}.txt")
                #o3d.io.write_point_cloud(output_file, rotated_point_cloud)
                
                # Update xyzw for next rotation
                xyzw = new_xyzw
                count += 1
            else:

                # Apply rotations and save the augmented files
                for angle in angles_to_rotate:
                
                    rotated_point_cloud = clone_pointcloud(point_cloud)

                    rotate_point_cloud(rotated_point_cloud, angle)
                    
                    # Formulate new filename based on max_xyzw + xyzw
                    new_xyzw = max_xyzw + count
                    #print(filename, new_xyzw)

                    #new_file_name
                    output_file = new_parent_directory + "/" + subfolder + "/" + f"{subfolder}_{new_xyzw}.txt"
                # print(output_file)
                    write_point_cloud(output_file, rotated_point_cloud)
                    #output_file = os.path.join(subfolder_path, f"{subfolder}{new_xyzw}.txt")
                    #o3d.io.write_point_cloud(output_file, rotated_point_cloud)
                    
                    # Update xyzw for next rotation
                    xyzw = new_xyzw
                    count += 1


assert(sanity == len(skip_files))