import os
from collections import defaultdict

import json
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.tv_tensors import Mask, BoundingBoxes

class RadioGalaxyNET(Dataset):
    def __init__(self, root: str, annFile: str, transforms=None):
        self.root, self.transforms = root, transforms
        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys())

        with open(annFile, 'r') as file:
            self.annotation = json.load(file)

        self.categories = {cat['id']: cat for cat in self.annotation['categories']}
        self.images = {img['id']: img for img in self.annotation['images']}
        
        self.segmentations = defaultdict(list)
        for ann in self.annotation['annotations']:
            imgId = ann['id']
            self.segmentations[imgId].append(ann)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        imgId = self.ids[idx] # corresponding imgId
        
        annIds = self.coco.getAnnIds(imgId)
        anns = self.coco.loadAnns(annIds)
        boxes, instanceMasks, labels, area = self.__annToTarget__(anns)
        semanticMasks = self.__instance2semantic__(instanceMasks, labels)
        
        pth = os.path.join(self.root, self.images[imgId]['filename'])
        img = read_image(pth)

        iscrowd = torch.zeros((len(anns),), dtype=torch.int64) # ?

        my_annotation = {'boxes': boxes, 
                         'masks': instanceMasks, 
                         'labels': labels,
                         'image_id': imgId, 
                         'area': area, 
                         'iscrowd': iscrowd}

        if self.transforms is not None: # this seems unfinished
            img = self.transforms(img)
        return img, my_annotation, semanticMasks
    
    def __annToTarget__(self, anns):
        bbox = [[ann['bbox'][0], ann['bbox'][1], 
                 ann['bbox'][0] + ann['bbox'][2], 
                 ann['bbox'][1] + ann['bbox'][3]] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        areas = [ann['area'] for ann in anns]
        masks = [self.coco.annToMask(ann) for ann in anns]

        bbox, masks = BoundingBoxes(bbox), Mask(masks)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        return bbox, masks, labels, areas

    def __instance2semantic__(self, instanceMasks, labels):
        h, w = 450, 450
        semanticMask = torch.zeros((h, w), dtype=torch.int64)
        for cat, mask in zip(labels, instanceMasks):
            semanticMask[mask == 1] = cat
        return semanticMask
    
    def __len__(self):
        return len(self.ids)
        
    def display_categories(self):
        for id, cat in self.categories.items():
            print(f'id {id}: {cat["name"]}')
        return None