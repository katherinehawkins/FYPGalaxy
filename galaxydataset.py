#Import all the packages

import os
import torch
from PIL import Image as PILImage
from torchvision import transforms
from torchvision import tv_tensors
import pathlib 
from io import BytesIO
from pycocotools.coco import COCO
import numpy as np
import base64
import json
from matplotlib import pyplot as plt

# Create the data-structure on our custom dataset.
# The template is created by Takashi Nakamura Modified by Yide To Incorporate Segmentation
# https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5

# Additional code for visualization are from Adam Kelly's code
#https://www.immersivelimit.com/course/creating-coco-datasets

class RadioGalaxyDataset(torch.utils.data.Dataset):
    
    # initialise function of class
    def __init__(self, root: str, annotation_path, transforms =None):
        # Define the root folder for all Images.
        self.root = root
        # Define the transformations which is to be applied on data.
        self.transforms = transforms
        # Define the coco Datastructure.
        self.coco = COCO(annotation_path)
        # Define the IDs
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Load Annotations Files (for display)
        json_file = open(annotation_path)
        self.annotation = json.load(json_file)
        json_file.close()

        # Define the four colors representing the four classes of the galaxy.
        self.colors = ['red', 'green', 'blue', 'yellow']

        # Return the number of classes from the Json file.
        self._process_categories()
        self._process_images()
        self._process_segmentations()
    ###################################################################################
    # Private Function: Core Get Item function.
    ###################################################################################
    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = PILImage.open(os.path.join(self.root, path))
        # Convert the image into torch vision tensor
        img = tv_tensors.Image(img)
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes

        # Generate masks used in the model
        masks = self._gen_masks(ann_ids)
        # Here mask variable is still underconstruction
        my_annotation["masks"] = tv_tensors.Mask(masks)
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        # Apply transformation onto our image.
        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    
    # Private function for generating the masks.
    def _gen_masks(self, ann_ids):
        mask_arr = []
        anns = self.coco.loadAnns(ann_ids)
        # Add background
        mask_back = self.coco.annToMask(anns[0])
        mask_arr.append(mask_back)
        for i in range(len(anns)):
            mask = self.coco.annToMask(anns[i])
            mask_arr.append(mask)
           #mask += self.coco.annToMask(anns[i])
        return np.array(mask_arr)

    ###################################################################################
    # Private Function: it will get the item from the dataset based on the given index
    ###################################################################################

    def load_image(self, index:int):
        # Opens an Image using PIL.
        img_name = self.images[index]['file_name']
        # Append the directory to the image path
        img_path = self.root+"/" + img_name
        return PILImage.open(img_path)
    
     ###################################################################################
    # Private Function: it will compute for the number of unique classes in the dataset.
    ###################################################################################
        
    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        for category in self.annotation['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')
            
            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id} # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}
 ###################################################################################
    # Private Function: Add all the Image Names to the list of names in the dataset.
    ###################################################################################
                
    def _process_images(self):
        self.images = dict()
        for image in self.annotation['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
    
       
    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.annotation['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)
    
    ###################################################################################
    # Public Function: Display all the classes in the Dataset.
    ###################################################################################
    def display_categories(self):
        print('Categories')
        print('==================')
        for sc_name, set_of_cat_ids in self.super_categories.items():
            print(f'  super_category: {sc_name}')
            for cat_id in set_of_cat_ids:
                print(f'    id {cat_id}: {self.categories[cat_id]["name"]}'
                     )
                
            print('')

    ###################################################################################
    # Public Function: Display an Image on a given Index.
    ###################################################################################
            
    def display_image(self, image_id, show_bbox=True, show_polys=True, show_crowds=True):
        print('Image')
        print('==================')
        
        # Print image info
        image = self.images[image_id]
        for key, val in image.items():
            print(f'  {key}: {val}')
            
        # Open the image
        image_path = pathlib.Path(self.root) / image['file_name']
        image = PILImage.open(image_path)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        data_uri = base64.b64encode(buffer.read()).decode('ascii')
        image_path = "data:image/png;base64,{0}".format(data_uri)
        
        # Calculate the size and adjusted display size
        max_width = 600
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height
        
        # Create bounding boxes and polygons
        bboxes = dict()
        polygons = dict()
        rle_regions = dict()
        seg_colors = dict()
        
        for i, seg in enumerate(self.segmentations[image_id]):
            if i < len(self.colors):
                seg_colors[seg['id']] = self.colors[i]
            else:
                seg_colors[seg['id']] = 'white'
                
            print(f'  {seg_colors[seg["id"]]}: {self.categories[seg["category_id"]]["name"]}')
            
            bboxes[seg['id']] = np.multiply(seg['bbox'], adjusted_ratio).astype(int)
            
            if seg['iscrowd'] == 0:
                polygons[seg['id']] = []
                for seg_points in seg['segmentation']:
                    seg_points = np.multiply(seg_points, adjusted_ratio).astype(int)
                    polygons[seg['id']].append(str(seg_points).lstrip('[').rstrip(']'))               
    
        
        # Draw the image
        html = '<div class="container" style="position:relative;">'
        html += f'<img src="{str(image_path)}" style="position:relative; top:0px; left:0px; width:{adjusted_width}px;">'
        html += '<div class="svgclass">'
        html += f'<svg width="{adjusted_width}" height="{adjusted_height}">'
        
        # Draw shapes on image
        if show_polys:
            for seg_id, points_list in polygons.items():
                for points in points_list:
                    html += f'<polygon points="{points}" \
                        style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0.5; stroke-width:1;" />'
        
        if show_crowds:
            for seg_id, line_list in rle_regions.items():
                for line in line_list:
                    html += f'<rect x="{line[0]}" y="{line[1]}" width="{line[2]}" height="{line[3]}" \
                        style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; \
                        fill-opacity:0.5; stroke-opacity:0.5" />'
        
        if show_bbox:
            for seg_id, bbox in bboxes.items():
                html += f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}" \
                    style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0" />'
        
        html += '</svg>'
        html += '</div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass {position: absolute; top:0px; left: 0px}'
        html += '</style>'
        
        return html
    