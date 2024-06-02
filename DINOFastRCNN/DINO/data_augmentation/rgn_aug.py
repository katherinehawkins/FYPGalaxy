import os
from PIL import Image
import numpy as np
import shutil

def load_and_split_images(source_folder, train_dest, val_dest, train_ratio=0.5):
    # Create destination folders if they don't exist
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    # List all files in the source directory
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    total_files = len(files)
    print(f"Total images found: {total_files}")
    
    # Calculate split index
    split_index = int(total_files * train_ratio)
    
    # Shuffle files randomly
    np.random.shuffle(files)
    
    # Split files
    train_files = files[:split_index]
    val_files = files[split_index:]
    
    # Function to process and save images
    def process_and_save(files, destination):
        for file in files:
            file_path = os.path.join(source_folder, file)
            # Open and convert to ensure RGB
            image = Image.open(file_path).convert('RGB')
            # Convert image to numpy array
            image_array = np.array(image)
            # Set the first channel (Red) to zero
            image_array[:, :, 0] = 0
            # Convert back to PIL Image and save
            new_image = Image.fromarray(image_array)
            new_image.save(os.path.join(destination, file))
    
    # Process and save training and validation images
    process_and_save(train_files, train_dest)
    process_and_save(val_files, val_dest)

    print("Images processed and saved into training and validation folders.")

# Set your paths
source_folder = 'data'
train_dest_folder = 'mirabest_aug2/train_dino'
val_dest_folder = 'mirabest_aug2/val_dino'

# Call the function
load_and_split_images(source_folder, train_dest_folder, val_dest_folder)