'''
This script uses open3d Api to add normals to the dataset. 

newFolderName - "Folder where we want point clouds with normals to be stores"
folderName - "Folder which contains original pointcloud dataset"

'''

import sys

from pathlib import Path
import numpy as np
import open3d as o3d
import laspy as lp
# Read the point cloud

folderName = sys.argv[1] #Get foldername from commandline
newFolderName = folderName +"_normal"
folderPath = Path(folderName)
# Get all the folders in the folder
folders = [f for f in folderPath.iterdir() if f.is_dir()]


#Get all the files in the folder
for folder in folders:
    files = [f for f in folder.iterdir() if f.is_file()]
    
    for file in files:

        file_name = str(file.absolute())

        # Check if the file exists
        if not Path(file_name).is_file():
            print("File does not exist:", file_name)
        else:
            # Try to read the point cloud
            try:
                pcd1 = o3d.io.read_point_cloud(file_name, format='xyz', print_progress=True)

               # o3d.visualization.draw_geometries([pcd1])
                print(pcd1)
                # Estimate normals
                pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))

                # Check if normals are successfully estimated
                if not pcd1.has_normals():
                    print("Normals were not successfully estimated.")
                else:
                    # Orient normals consistently
                    pcd1.orient_normals_consistent_tangent_plane(k=15)


                    newFolder = Path(newFolderName + "/" + folder.name)
                    newFolder.mkdir(parents=True, exist_ok=True)
                    #new_file_name
                    newFileName = newFolderName + "/" + folder.name + "/" + file.name
                   
                    # Open the output file for writing in 'xyzn' format
                    with open(newFileName, 'w') as output_file:
                        # Iterate through the points in the point cloud
                        for i in range(len(pcd1.points)):
                            point = pcd1.points[i]
                            normal = pcd1.normals[i]

                            output_file.write(f"{point[0]},{point[1]},{point[2]},{normal[0]},{normal[1]},{normal[2]}\n")

                    print(f"Point cloud with normals written to '{newFileName}' in 'xyzn' format.")
            except Exception as e:
                print("An error occurred:", str(e))

 

        
