import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from base_dataset import RadioGalaxyNET

class SAMDataset(Dataset):
    def __init__(self, root, annFile, processor, max_boxes=5, transform=None, transforms=None):
        self.dataset = RadioGalaxyNET(root, annFile, transform, transforms)
        self.processor, self.max_boxes = processor, max_boxes

    def __getitem__(self, idx):
        img, ann = self.dataset[idx]
        
        boxes, masks, labels = self.__pad__(ann['boxes'], ann['masks'])
        boxes, labels = [[boxes.tolist()]], [labels.tolist()]

        input = self.processor(img, 
                               input_boxes=boxes, 
                               input_labels=labels,
                               return_tensors="pt")
        
        input = {k : torch.squeeze(v, dim=0) for k,v in input.items()}
        input["ground_truth_mask"] = masks
        return input
        
    def __pad__(self, boxes, masks):
        to_pad = self.max_boxes - len(boxes)

        labels = np.ones((self.max_boxes,))
        labels[to_pad:] = -10

        boxes = F.pad(boxes, (0,0,0,to_pad), value=0)
        masks = F.pad(masks, (0,0,0,0,0,to_pad), value=0)
        return boxes, masks, labels

    def __len__(self):
        return len(self.dataset)