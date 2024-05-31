import json
import os
import numpy as np

# Function to convert segmentation mask to binary masks for each class
def convert_to_binary_masks(mask_data, num_classes):
    binary_masks = []
    for class_id in range(num_classes):
        binary_mask = (mask_data == class_id).astype(np.uint8)
        binary_masks.append(binary_mask)
    return binary_masks

# Path to the JSON file containing all the masks
json_file_path = 'runs/segment/val8/predictions.json'
# Path to the directory where ground truth text files will be saved
gt_dir = '../datasets/RadioGalaxyNET/data/RadioGalaxyNETSeg/labels/test'
final_dir = 'runs/segment/converted_json_GAL1'

num_classes = 4

# Load JSON file containing all the masks
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Initialize a set to store unique class IDs
class_ids = set()

# Iterate through subfolders under 'root'
for mask in data:
    # Access the data within the current subfolder
    image_id = mask['image_id']
    category_id = mask['category_id']
    segmentation_mask = mask['segmentation']
    
    # Convert segmentation mask to binary masks for each class
    binary_masks = convert_to_binary_masks(segmentation_mask, num_classes)
    
    # Create a new text file for each image
    txt_filename = os.path.splitext(image_id)[0] + '.txt'
    txt_filepath = os.path.join(final_dir, txt_filename)
    
    # Save binary masks to the text file
    with open(txt_filepath, 'w') as txt_file:
        for binary_mask in binary_masks:
            # Convert the binary mask to a string representation
            mask_str = ' '.join(map(str, binary_mask.flatten().tolist()))
            txt_file.write(f"{mask_str}\n")

