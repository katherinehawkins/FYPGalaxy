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
        self.root = root
        self.transform = transform
        self.transforms = transforms
        self.h, self.w = 450, 450

        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys()) # image_ids
        self.__createIndex__(annFile)

    def __createIndex__(self, annFile):
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
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        imgId = self.ids[idx] # corresponding imgId
        img, boxes, instanceMasks, labels, iscrowd = self.__id2item__(imgId)
        img, boxes, instanceMasks, area = self.__transform__(img, boxes, instanceMasks)
        return self.__formatOutput__(imgId, img, boxes, instanceMasks, labels, iscrowd, area)
    
    def __formatOutput__(self, imgId, img, boxes, instanceMasks, labels, iscrowd, area):
        my_annotation = {'boxes': boxes, # typical for detection
                         'masks': instanceMasks, # rewrite after inheritance
                         'labels': labels,
                         'image_id': imgId, 
                         'area': area, 
                         'iscrowd': iscrowd}
        return img, my_annotation
        
    def __transform__(self, img, boxes, instanceMasks):
        if self.transforms is not None:
            img, boxes, instanceMasks = self.transforms(img, boxes, instanceMasks)
        
        if self.transform is not None:
            img = self.transform(img)
        
        area = box_area(boxes) # recompute area after transform
        return img, boxes, instanceMasks, area

    def __id2item__(self, imgId):
        file_pth = os.path.join(self.root, self.images[imgId]['file_name'])
        img = read_image(file_pth)

        annIds = self.coco.getAnnIds(imgId)
        anns = self.coco.loadAnns(annIds)
        boxes, instanceMasks, labels = self.__annToTarget__(anns)
        
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64) # ??
        return img, boxes, instanceMasks, labels, iscrowd
    
    def __annToTarget__(self, anns):
        bbox = [[ann['bbox'][0], ann['bbox'][1], # converts from XYWH to XYXY
                 ann['bbox'][0] + ann['bbox'][2], 
                 ann['bbox'][1] + ann['bbox'][3]] for ann in anns]
        bbox = BoundingBoxes(bbox, format='XYXY', canvas_size=(self.h, self.w))
        
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        masks = Mask(np.array([self.coco.annToMask(ann) for ann in anns]))
        return bbox, masks, labels

    def __instance2semantic__(self, instanceMasks, labels):
        semanticMask = np.zeros((self.h, self.w), dtype=np.int64)
        for cat, mask in zip(labels, instanceMasks):
            semanticMask[mask == 1] = cat
        return Mask(semanticMask) # rewrite in inheritance if needed
    
    def __len__(self):
        return len(self.ids)
        
    def display_categories(self):
        for id, cat in self.categories.items():
            print(f'id {id}: {cat["name"]}')
        return None

        
