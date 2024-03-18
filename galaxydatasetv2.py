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
    def __init__(self, root: str, annFile: str, detection=False, transform=None, transforms=None):
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.detection = detection

        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys())
        self.__createIndex__(annFile)

    def __createIndex__(self, annFile):
        with open(annFile, 'r') as file:
            self.annotation = json.load(file)

        self.categories = {cat['id']: cat for cat in self.annotation['categories']}
        self.images = {img['id']: img for img in self.annotation['images']}
        
        self.segmentations = defaultdict(list)
        for ann in self.annotation['annotations']:
            imgId = ann['id']
            self.segmentations[imgId].append(ann)
        return None

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        
        imgId = self.ids[idx] # corresponding imgId
        file_pth = os.path.join(self.root, self.images[imgId]['file_name'])
        img = read_image(file_pth)

        annIds = self.coco.getAnnIds(imgId)
        anns = self.coco.loadAnns(annIds)
        boxes, instanceMasks, labels, area = self.__annToTarget__(anns)
        semanticMask = self.__instance2semantic__(instanceMasks, labels)

        if self.transforms is not None:
            img, boxes, instanceMasks, semanticMask = self.transforms(img, boxes, instanceMasks, semanticMask)
            area = box_area(boxes)
        
        if self.transform is not None:
            img = self.transform(img)

        iscrowd = torch.zeros((len(anns),), dtype=torch.int64) # ?
        my_annotation = {'boxes': boxes, 
                         'masks': instanceMasks, 
                         'labels': labels,
                         'image_id': imgId, 
                         'area': area, 
                         'iscrowd': iscrowd}
        
        return (img, my_annotation) if self.detection else (img, semanticMask)
    
    def __annToTarget__(self, anns):
        bbox = [[ann['bbox'][0], ann['bbox'][1], 
                 ann['bbox'][0] + ann['bbox'][2], 
                 ann['bbox'][1] + ann['bbox'][3]] for ann in anns]
        # converts from XYWH to XYXY
        
        labels = [ann['category_id'] for ann in anns]
        areas = [ann['area'] for ann in anns]
        masks = np.array([self.coco.annToMask(ann) for ann in anns])

        bbox = BoundingBoxes(bbox, format='XYXY', canvas_size=(450, 450))
        masks, labels = Mask(masks), torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        return bbox, masks, labels, areas

    def __instance2semantic__(self, instanceMasks, labels):
        h, w = 450, 450
        semanticMask = np.zeros((h, w), dtype=np.int64)
        for cat, mask in zip(labels, instanceMasks):
            semanticMask[mask == 1] = cat
        return Mask(semanticMask)
    
    def __len__(self):
        return len(self.ids)
        
    def display_categories(self):
        for id, cat in self.categories.items():
            print(f'id {id}: {cat["name"]}')
        return None