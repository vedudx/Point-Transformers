from pathlib import Path
import numpy as np
import open3d as o3d
import laspy as lp
# Read the point cloud
folderName = "new_dataset"
newFolderName = "new_dataset_normal"
folderPath = Path(folderName)
# Get all the folders in the folder
folders = [f for f in folderPath.iterdir() if f.is_dir()]


#Get all the files in the folder
for folder in folders:
    files = [f for f in folder.iterdir() if f.is_file()]
    
    for file in files:

        file_name = str(file.absolute())

        # file = open(file_name, 'r')
        # data = print(file.read())


        # sample_pcd_data = o3d.data.PCDPointCloud()
        # print(type(sample_pcd_data.path), sample_pcd_data.path)
        # pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
        # print(pcd)

        # file_name3="/run/media/vedu/Extreme SSD/modelnet40_normal_resampled/night_stand/night_stand_0020.txt"
        # file_name_n="/run/media/vedu/Extreme SSD/modelnet40_new_normal_resampled/night_stand/night_stand_0020.txt"
        # file_name2 = "/run/media/vedu/Extreme SSD/modelnet40_normal_resampled/airplane/airplane_0001.txt"
        #file_name = '/run/media/vedu/Extreme SSD/new_dataset/larch/larch_0030.txt'



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

                    # Visualize the point cloud with normals
                    #o3d.visualization.draw_geometries([pcd1], point_show_normal=True)

                         #create new folder
                    newFolder = Path(newFolderName + "/" + folder.name)
                    newFolder.mkdir(parents=True, exist_ok=True)
                    #new_file_name
                    newFileName = newFolderName + "/" + folder.name + "/" + file.name
                    #o3d.io.write_point_cloud(newFile, pcd1, print_progress=True)

                                        # Open the output file for writing in 'xyz' format
                   
                    # Open the output file for writing in 'xyzn' format
                    with open(newFileName, 'w') as output_file:
                        # Iterate through the points in the point cloud
                        for i in range(len(pcd1.points)):
                            point = pcd1.points[i]
                            normal = pcd1.normals[i]
                            #print(type(point[0]), type(normal[0]))
                            # Write the X, Y, Z, Nx, Ny, Nz coordinates separated by spaces
                            output_file.write(f"{point[0]},{point[1]},{point[2]},{normal[0]},{normal[1]},{normal[2]}\n")

                    print(f"Point cloud with normals written to '{newFileName}' in 'xyzn' format.")
            except Exception as e:
                print("An error occurred:", str(e))

 

        
        #file_name='/run/media/vedu/Extreme SSD/dataset/larch/larch_0030.txt'
        # filename2='/run/media/vedu/Extreme SSD/607468_5907153_aoi_ht.las/good/607468_5907153_aoi_ht.las_00001.pcd'
        #pcd = o3d.io.read_point_cloud(filename, format='xyzn', print_progress=True)

        # pcd1 = o3d.io.read_point_cloud(file_name2, format='xyz', print_progress=True)

        # pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))
        # # pcd2 = o3d.io.read_point_cloud(filename2)
        # pcd1.orient_normals_consistent_tangent_plane(k=15)
        # o3d.visualization.draw_geometries([pcd1], point_show_normal=True)

        #print(np.linalg.norm(np.asarray(pcd1.normals)-np.asarray(pcd.normals)))
        #print(np.asarray(pcd1.normals))


      
    #     o3d.visualization.draw_geometries([pcd1], point_show_normal=True)

    #     pcd2 = o3d.io.read_point_cloud(file_name, format='xyzn', print_progress=True)
    #     #pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))
    #     # pcd2 = o3d.io.read_point_cloud(filename2)
    #    # pcd2.orient_normals_consistent_tangent_plane(k=15)
    #     o3d.visualization.draw_geometries([pcd2], point_show_normal=True)



      
        #calculate average distance between pcd1 normals and pcd normals

        # o3d.visualization.draw_geometries([pcd1], point_show_normal=True)
 
        #f.close()


   
        # Write the file
        # newFile = open(newFolderName + "/" + folder.name + "/" + file.name, "w")
        # for line in lines:
        #     newFile.write(line.replace("normal", "xyz"))
        # newFile.close()

    