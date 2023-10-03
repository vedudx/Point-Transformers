from pathlib import Path

folderName = "dataset"
newFolderName = "giga_dataset"
folderPath = Path(folderName)
folderPath.mkdir(parents=True, exist_ok=True)

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

