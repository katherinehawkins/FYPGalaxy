#----------------------------------------------------------------------------------------------------
# Import Packages
#----------------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
import torchvision
from CRUMB import CRUMB
import torchvision.transforms as transforms
import os
import random
import torch
from PIL import Image

#----------------------------------------------------------------------------------------------------
# Define Dataset and Augmentation
#----------------------------------------------------------------------------------------------------
batch_size = 80
imsize = 150 # this value is fixed
crop = transforms.CenterCrop(imsize)
rotate = transforms.RandomRotation([-180, 180])
totensor = transforms.ToTensor()
normalise = transforms.Normalize((0.0029,), (0.0341,)) # CRUMB mean and stdev

transforms_mb = transforms.Compose([
    crop,
    rotate,
    totensor,
    normalise
])

# load training and test set
# this will download CRUMB to a directory called "crumb"
test_data = CRUMB('crumb', download=True, train=False, transform=transforms_mb)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
train_data = CRUMB('crumb', download=True, train=True, transform=transforms_mb)
train_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

#----------------------------------------------------------------------------------------------------
# Define Random Seed
#----------------------------------------------------------------------------------------------------
random.seed(42)
torch.manual_seed(42)

# Build the directory for storing the simulated Data
directory = 'mirabest_aug2/train'
if not os.path.exists(directory):
    os.makedirs(directory)

#----------------------------------------------------------------------------------------------------
# Define Function to Randomly Place the Image Within the Map
#----------------------------------------------------------------------------------------------------
def place_images(galaxy_tensors, num_images, size=(450, 450), possible_sizes=[(100, 100), (150, 150), (200, 200), (300, 300), (350, 350)]):
    # Create a blank RGB image of 450x450
    large_image = torch.zeros((3, size[0], size[1]))

     # Define transformations
    rotate_transform = transforms.RandomRotation(degrees=(0, 360))
    #resize_transforms = [transforms.Resize(size=s) for s in possible_sizes]

    for _ in range(num_images):
        # Randomly select a galaxy tensor
        image_tensor = random.choice(galaxy_tensors)

        # Randomly rotate the image
        image_tensor = rotate_transform(image_tensor)
        
        # Randomly choose a size for the small image
        small_size = random.choice(possible_sizes)

        # Randomly generate x, y coordinates for the top-left corner of the small image
        x = random.randint(0, size[0] - small_size[0])
        y = random.randint(0, size[1] - small_size[1])

        # Resize the image tensor to the chosen small size
        resized_image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=small_size, mode='bilinear', align_corners=False).squeeze(0)

        # Find non-zero points in the resized tensor
        non_zero_indices = torch.nonzero(resized_image_tensor, as_tuple=False)

        # Shuffle and split indices into two halves
        shuffled_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))]
        half_point = len(shuffled_indices) // 2

        # Get the indices for the green and blue channels
        green_indices = shuffled_indices[:half_point]
        blue_indices = shuffled_indices[half_point:]

        #print(green_indices)
        # Place the selected non-zero values in the green channel
        for idx in green_indices:
            #print(idx[1])
            #print(idx[2])
            #print(resized_image_tensor.shape)
            large_image[1, x + idx[1], y + idx[2]] = resized_image_tensor[0, idx[1], idx[2]]

        # Place the selected non-zero values in the blue channel
        for idx in blue_indices:
            large_image[2, x + idx[1], y + idx[2]] = resized_image_tensor[0, idx[1], idx[2]]

    return large_image

#----------------------------------------------------------------------------------------------------
# Define Function to Add simulated Infred Channel
#----------------------------------------------------------------------------------------------------
def add_infred(add_inf, large_image_tensor):
    if add_inf == True:
        blue_and_green_average = (large_image_tensor[1] + large_image_tensor[2])/2

        non_zero_indices = np.where(blue_and_green_average != 0)
        # Set the kernel size and standard deviation for the Gaussian blur
        kernel_size = 21  # Adjust the size to control the blur effect
        sigma = 2  # Adjust sigma to control the spread of the blur

            
        sample_size = 50
        if len(non_zero_indices[0]) > sample_size:
            selected_indices = torch.randperm(len(non_zero_indices[0]))[:sample_size]
            selected_points = (non_zero_indices[0][selected_indices], non_zero_indices[1][selected_indices])
            large_image_tensor[0][selected_points] = 10
        else:
            # If fewer than 500 non-zero points exist, just modify all of them
            large_image_tensor[0][non_zero_indices] = 10

        # Apply Gaussian blur to the red channel
        # Add padding to maintain the size of the image after applying the kernel
        padding = kernel_size // 2
        #red_channel_padded = F.pad(large_image_tensor[0].unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
        blurred_red_channel = torchvision.transforms.functional.gaussian_blur(large_image_tensor[0].unsqueeze(0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        large_image_tensor[0] = blurred_red_channel.squeeze() 

        block_size = 10
        noise_std = 0.03
        # Calculate the dimensions for the smaller noise tensor
        small_noise_dims = (large_image_tensor.size(1) // block_size, large_image_tensor.size(2) // block_size)
        small_noise = torch.randn((small_noise_dims)) * noise_std
        # Upscale the noise to the original image size
        large_noise = torch.nn.functional.interpolate(small_noise.unsqueeze(0).unsqueeze(0), 
                                                        size=(large_image_tensor.size(1), large_image_tensor.size(2)), 
                                                        mode='nearest').squeeze(0)

        # Add the upscaled noise to the red channel
        large_image_tensor[0] += 5*large_noise.squeeze()
        large_image_tensor[0] = torch.clamp(large_image_tensor[0], 0, 1)  # Clamp values to maintain valid pixel range
    else:
        large_image_tensor = large_image_tensor
    return large_image_tensor

def place_1_gal(inverted_gray):
# Randomly allocate the points into either green or blue channels
    non_zero_indices = torch.nonzero(inverted_gray, as_tuple=False)
    # Shuffle and split indices into two halves
    shuffled_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))]
    half_point = len(shuffled_indices) // 2
    # Get the indices for the green and blue channels
    image_temp = torch.zeros((3, 450, 450))

    green_indices = shuffled_indices[:half_point]
    blue_indices = shuffled_indices[half_point:]

    for idx in green_indices:
        image_temp[1, idx[1], idx[2]] = inverted_gray[0, idx[1], idx[2]]
    # Place the selected non-zero values in the blue channel
    for idx in blue_indices:
        image_temp[2, idx[1], idx[2]] = inverted_gray[0, idx[1], idx[2]]

    return image_temp
#----------------------------------------------------------------------------------------------------
# Simulation Loop
#----------------------------------------------------------------------------------------------------
total_images = 10000  # total images to generate
galaxy_tensors = [data[0].cpu() for data in train_data]  # Pre-load all galaxy tensors

for index in range(total_images):
    if index < len(galaxy_tensors):
        # Select the tensor and resize
        image_tensor = galaxy_tensors[index]
        if image_tensor.dim() == 2 or (image_tensor.dim() == 3 and image_tensor.shape[0] == 1):
        # Expand or repeat the grayscale image to 3 channels
            image_tensor[np.where(image_tensor < 0)] = 1        
            inverted_gray = 1.0 - image_tensor
            

        elif image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            # Already in 3 channels, do nothing
            pass
        else:
            raise ValueError("Unsupported channel size: {}".format(image_tensor.shape[0]))
        
        # Resize the Image
        large_image_tensor = torchvision.transforms.functional.resize(inverted_gray, (450, 450))
        large_image_tensor = place_1_gal(large_image_tensor)
        
    
        #print(large_image_tensor.shape)
    else:
        # if index < size  
        num_small_images = random.randint(2, 7)

        # Select a random subset of galaxy images to use
        selected_galaxies = random.sample(galaxy_tensors, num_small_images)

        # Ensure tensors have no negative values and invert them
        selected_galaxies = [torch.where(tensor < 0, torch.ones_like(tensor), tensor) for tensor in selected_galaxies]

        # Now perform the inversion by subtracting each tensor from 1
        selected_galaxies = [1 - tensor for tensor in selected_galaxies]

        # Place the images
        large_image_tensor = place_images(selected_galaxies, num_small_images)
        
        large_image_tensor = add_infred(False, large_image_tensor)
    
    # Convert to PIL Image
    image = Image.fromarray((large_image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    print(index)
    # Define the filename with the path, using the index
    filename = os.path.join(directory, f"{index}.png")
    image.save(filename)  # Save the image as a PNG file

print("Image generation complete.")