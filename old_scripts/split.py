import os
import random

# Define the path to the dataset
parent_folder = "giga_dataset"

# Define the names of the output text files
train_file_name = parent_folder+"/modelnet5_train.txt"
test_file_name = parent_folder+"/modelnet5_test.txt"

# Open the output text files in write mode
with open(train_file_name, 'w') as train_file, open(test_file_name, 'w') as test_file:
    
    # Loop through each subfolder in the parent folder
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)

        # Check if it is a directory/folder
        if os.path.isdir(folder_path):

            # List all files in the subfolder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # Shuffle the file list
            random.shuffle(files)

            # Split the files into train and test with an 80:20 ratio
            train_len = int(0.85 * len(files))
            train_files = files[:train_len]
            test_files = files[train_len:]

            # Write the names of the train and test files to the respective text files
            for file in train_files:
                train_file.write(str(file).replace(".txt", "") + '\n')

            for file in test_files:
                test_file.write(str(file).replace(".txt", "") + '\n')

print("Train and Test file lists are created successfully!")
