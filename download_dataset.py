import shutil
import os
import kagglehub

# Create a destination directory (change this to your preferred folder)
destinationFolder = "my_data"

# Make the folder if it doesn’t exist
os.makedirs(destinationFolder, exist_ok=True)

# Download the dataset
datasetPath = kagglehub.dataset_download("wyattowalsh/basketball")

shutil.copytree(datasetPath, destinationFolder, dirs_exist_ok=True)
