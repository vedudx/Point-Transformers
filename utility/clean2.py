import sys
from pathlib import Path
import shutil

folderName=""
newFolderName=""
# Get folder names from command line arguments
if len(sys.argv) < 2:
    print("Usage: script.py <source_folder> <destination_folder>")
    sys.exit(1)
elif len(sys.argv) == 2:
    print("one folder name shared so overwriting....")
    folderName = sys.argv[1]
    #create a temp folder and then later overwrite folderName
    newFolderName = folderName + "_temp"
else:
    folderName = sys.argv[1]
    newFolderName = sys.argv[2]



# Create the source and destination folders if they don't exist
folderPath = Path(folderName)
folderPath.mkdir(parents=True, exist_ok=True)
newFolderPath = Path(newFolderName)
newFolderPath.mkdir(parents=True, exist_ok=True)

# Get all the folders in the folder
folders = [f for f in folderPath.iterdir() if f.is_dir()]

# Iterate over each folder and its files
for folder in folders:
    newFolder = Path(newFolderName) / folder.name
    newFolder.mkdir(parents=True, exist_ok=True)
    
    # Get all the files in the folder
    files = [f for f in folder.iterdir() if f.is_file()]
    for file in files:
        # Read the file
        with open(file, "r") as f:
            lines = f.readlines()

        # Write the file
        newFilePath = newFolder / file.name
        with open(newFilePath, "w") as newFile:
            for line in lines:
                newFile.write(line.replace(",", " "))

# If only one folder name was provided, replace the original folder with the temp folder
if len(sys.argv) == 2:
    shutil.rmtree(folderPath)
    shutil.move(newFolderPath, folderPath)