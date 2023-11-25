import os
import shutil

# LAST

source_dir = 'path/to/LAST'  # Path to the original directory
dest_dir = 'path/to/new_folder'  # Path to the new directory

for root, dirs, files in os.walk(source_dir):
    for name in files:
        file_path = os.path.join(root, name)
        relative_path = os.path.relpath(root, source_dir)
        relative_path = relative_path.split('/')[-1]
        
        # Check if the directory name follows a specific format (e.g., '000002')
        if relative_path.isdigit():
            new_dir = os.path.join(dest_dir, relative_path)

            # Create a new directory if needed
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            # Copy the file
            shutil.copy(file_path, new_dir)
            

# Celeb-reID

base_dir = 'path/to/Celeb-reID'  # Path to the Celeb-reID folder
subfolders = ['gallery', 'query']  # List of subfolders
new_dir = 'path/to/new_folder' # Path to the new directory

for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    
    for file in os.listdir(subfolder_path):
        if file.endswith(".jpg"):  # Ensure the file is a JPEG image
            file_path = os.path.join(subfolder_path, file)
            folder_name = file.split('_')[0].zfill(3)  # Get the first number from the filename and format it as a three-digit number

            new_dir = os.path.join(base_dir, folder_name)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            shutil.copy(file_path, new_dir)  # Copy the file to the new directory

