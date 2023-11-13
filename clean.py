from pathlib import Path
import numpy as np
import open3d as o3d
import laspy as lp
# Read the point cloud
folderName = "giga_dataset_normal/"
newFolderName = "giga_dataset_normal_augment/"
folderPath = Path(folderName)
folderPath.mkdir(parents=True, exist_ok=True)
# Get all the folders in the folder
folders = [f for f in folderPath.iterdir() if f.is_dir()]

#Get all the files in the folder
for folder in folders:
    files = [f for f in folder.iterdir() if f.is_file()]
    for file in files:
        # Read the file
        f = open(file, "r")
        lines = f.readlines()
        # print(lines)

        f.close()
    
        #create new folder
        newFolder = Path(newFolderName + "/" + folder.name)
        newFolder.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        newFile = open(newFolderName + "/" + folder.name + "/" + file.name, "w")
        for line in lines:
            newFile.write(line.replace(",", " "))
        newFile.close()

        # break
   

