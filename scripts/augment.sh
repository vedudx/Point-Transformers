o_folder=$1
n_folder=$2

python3 ../utility/clean2.py "$o_folder" "$n_folder"

python3 ../utility/normal.py "$n_folder"

# Correctly concatenate strings to form a new folder name
n_folder_2="${n_folder}_normal"


#Need to clean files again
python3 ../utility/clean2.py "$n_folder_2" 

#Now we augment the folder with new pointclouds
