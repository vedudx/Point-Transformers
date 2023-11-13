import os
import argparse

def count_files_in_directory(directory):
    count = 0
    for foldername, subfolders, filenames in os.walk(directory):
        # Count only files, not subfolders
        count += len(filenames)
    return count

def main(parent, o):
    # Replace with the path of your parent directory
    parent_directory = parent
    
    # Specify the output file
    output_file = o
    total_count = 0
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Go through each folder inside the parent directory
        for foldername, subfolders, filenames in os.walk(parent_directory):
            # Avoid counting the files in the parent_directory itself
            if foldername == parent_directory:
                continue
            
            # Count the number of files in the folder
            num_files = count_files_in_directory(foldername)

            total_count += num_files
            
            # Write the folder name and the number of files to the output file
            f.write(f'{foldername}: {num_files} files\n')
            print(f'{foldername}: {num_files} files')
        # Write the folder name and the number of files to the output file
        f.write(f'{parent_directory}: {total_count} files\n')
        print(f'{parent_directory}: {total_count} files\n')
if __name__ == "__main__":
      # Add arguments
    parser = argparse.ArgumentParser(description='Count the number of files in each folder inside a parent directory.')

    parser.add_argument('-parent', type=str, help='The path of the parent directory to analyze.')
    parser.add_argument('-o', '--output', type=str, default='output.txt', help='The path of the output file to write to.')
      # Parse the arguments
    args = parser.parse_args()
    
    main(args.parent, args.output)
