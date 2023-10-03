import random
from pathlib import Path
import numpy as np

# List of filenames



folderName = "new_dataset_normal"
filenames = []

folderPath = Path(folderName)
# Get all the folders in the folder
folders = [f for f in folderPath.iterdir() if f.is_dir()]

with open(folderName+'/modelnet5_train.txt', 'w') as train_file:
    pass

with open(folderName+ '/modelnet5_val.txt', 'w') as val_file:
    pass

with open(folderName+ '/modelnet5_test.txt', 'w') as test_file:
    pass

print("Files written successfully.")

#Get all the files in the folder
for folder in folders:
    files = [f for f in folder.iterdir() if f.is_file()]
    filenames.clear()
    for file in files:
        #print(file, type(file), str(file), str(file.absolute()))
        print(str(file.absolute()), str(file), file.name[:-4])
        filenames.append(file.name[:-4])
# Calculate the number of filenames for each split
    total_filenames = len(filenames)
    train_count = int(total_filenames * 0.85)
    #val_count = int(total_filenames * 0.1)
    val_count = 0
    test_count = total_filenames - (train_count + val_count)

    # Randomly shuffle the list of filenames
    random.shuffle(filenames)

    # Split the filenames into train, val, and test sets
    train_filenames = filenames[:train_count]
    val_filenames = filenames[train_count:train_count+val_count]
    test_filenames = filenames[train_count+val_count:]

    print(test_filenames)

    # Write the train, val, and test filenames to their respective files
    with open(folderName+'/modelnet5_train.txt', 'a') as train_file:
        train_file.write('\n'.join(train_filenames))

    with open(folderName+ '/modelnet5_val.txt', 'a') as val_file:
        val_file.write('\n'.join(val_filenames))

    with open(folderName+ '/modelnet5_test.txt', 'a') as test_file:
        test_file.write('\n'.join(test_filenames))

    print("Files written successfully.")
