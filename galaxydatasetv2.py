import os
from collections import defaultdict

import json
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_area
from torchvision.tv_tensors import Mask, BoundingBoxes

class RadioGalaxyNET(Dataset):
    def __init__(self, root: str, annFile: str, transform=None, transforms=None):
        """
        Base dataset class for RadioGalaxyNET.

        Args:
            root (str): Root directory containing the images.
            annFile (str): Path to the annotation file in COCO format.
            transform (callable, optional): A function/transform to apply to the image.
            transforms (callable, optional): A function/transform to apply to the image, boxes, and masks.

        """
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.h, self.w = 450, 450

        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys()) # image_ids
        self.__createIndex__(annFile)

    def __createIndex__(self, annFile):
        """
        Creates dictionaries that map image_id to annotations and image information.

        Args:
            annFile (str): Path to the annotation file in COCO format.

        """
        with open(annFile, 'r') as file:
            self.annotation = json.load(file)

        self.categories = {cat['id']: cat for cat in self.annotation['categories']}
        self.images = {img['id']: img for img in self.annotation['images']}
        
        self.segmentations = defaultdict(list) # maps image_id to annotations
        for ann in self.annotation['annotations']:
            imgId = ann['id']
            self.segmentations[imgId].append(ann)
        return None

    def __getitem__(self, idx):
        """
        Get item from dataset at specified index (not image_id).

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple with image, boxes, masks, etc. formatted according to __formatOutput__.

        """
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        imgId = self.ids[idx] # corresponding imgId
        img, boxes, instanceMasks, labels, iscrowd, area = self.__id2item__(imgId)
        img, my_annotation = self.__formatOutput__(imgId, img, boxes, instanceMasks, labels, iscrowd, area) # [Mod: Yide 20/03/2024]: transform requires my_annotation as input
        img, my_annotation = self.__transform__(img, my_annotation)

        return img, my_annotation
    
    def __formatOutput__(self, imgId, img, boxes, instanceMasks, labels, iscrowd, area):
        """
        Format the output for an item in the dataset.
        Please re-define after inheriting this class according to usecase.

        Args:
            imgId (int): Image ID.
            img (Tensor): Image tensor of shape (3, 450, 450).
            boxes (Tensor): Bounding boxes of shape (N, 4).
            instanceMasks (Tensor): Instance masks of shape (N, 450, 450).
            labels (Tensor): Labels for each box / mask of shape (N).
            iscrowd (Tensor): Indicates whether the object is a crowd. Shape (N).
            area (Tensor): Area of the bounding boxes. Shape (N).

        Returns:
            tuple: A tuple containing the image and its annotation.

        """
        my_annotation = {'boxes': boxes, # typical for detection
                         'masks': instanceMasks, # rewrite after inheritance
                         'labels': labels,
                         'image_id': imgId, 
                         'area': area, 
                         'iscrowd': iscrowd}
        return img, my_annotation
        
    def __transform__(self, img, annotation):
        """
        Apply transformations to the image, boxes, and masks.

        Args:
            img (Tensor): Image tensor of shape (3, 450, 450).
            boxes (Tensor): Bounding boxes of shape (N, 4).
            instanceMasks (Tensor): Instance masks of shape (N, 450, 450).

        Returns:
            tuple: A tuple containing the transformed image, boxes, masks, and area.

        """
        if self.transforms is not None:
            img, annotation = self.transforms(img, annotation)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, annotation

    def __id2item__(self, imgId):
        """
        Map image ID to item in the dataset.
        Returns tensors that are ready to be transformed and formatted.

        Args:
            imgId (int): Image ID.

        Returns:
            tuple: A tuple of tensors containing the image, boxes, masks, labels, and iscrowd.

        """
        file_pth = os.path.join(self.root, self.images[imgId]['file_name'])
        img = read_image(file_pth)

        annIds = self.coco.getAnnIds(imgId)
        anns = self.coco.loadAnns(annIds)
        boxes, instanceMasks, labels = self.__annToTarget__(anns)
        
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64) #  [Mod: Yide 20/03/2024]: iscrowd not important for our case
        area = box_area(boxes)  #  [Mod: Yide 20/03/2024]: added area calculation here
        return img, boxes, instanceMasks, labels, iscrowd, area
    
    def __annToTarget__(self, anns):
        """
        Interprets boxes, masks, and labels from annotations in the dataset.

        Args:
            anns (list): List of annotations.

        Returns:
            tuple: A tuple of tensors containing bounding boxes, masks, and labels.

        """
        bbox = [[ann['bbox'][0], ann['bbox'][1], # converts from XYWH to XYXY
                 ann['bbox'][0] + ann['bbox'][2], 
                 ann['bbox'][1] + ann['bbox'][3]] for ann in anns]
        bbox = BoundingBoxes(bbox, format='XYXY', canvas_size=(self.h, self.w))
        
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        masks = Mask(np.array([self.coco.annToMask(ann) for ann in anns]))
        return bbox, masks, labels

    def __instance2semantic__(self, instanceMasks, labels):
        """
        Convert instance mask tensors to semantic mask tensors.

        Args:
            instanceMasks (Tensor): Instance masks of shape (N, 450, 450).
            labels (Tensor): Labels of shape (N).

        Returns:
            Mask: Semantic mask tensor of shape (450, 450) with the corresponding category number of each pixel in each element.

        """
        semanticMask = np.zeros((self.h, self.w), dtype=np.int64)
        for cat, mask in zip(labels, instanceMasks):
            semanticMask[mask == 1] = cat
        return Mask(semanticMask) # rewrite in inheritance if needed
    
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.ids)
        
    def display_categories(self):
        """
        Display categories in the dataset.

        """
        for id, cat in self.categories.items():
            print(f'id {id}: {cat["name"]}')
        return None
    
    def display_image(self, idx):
        """
        Display Image with instance segmentation mask
        """
    


        
