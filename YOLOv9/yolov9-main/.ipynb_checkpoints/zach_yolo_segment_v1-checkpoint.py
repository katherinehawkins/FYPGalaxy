import os
import argparse
import json
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
from tqdm import tqdm
import subprocess

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)



print("ATTEMPTING TO INSTALL REQUIREMENTS:")
#os.system("pip install -r requirements.txt")
#os.system("pip install ultralytics --upgrade")



from ultralytics import YOLO

#TRAINING YOLOv9
# Initialize the YOLO model
#model = YOLO('models/segment/yolov9c-seg.yaml', task='segment')
#results = model.train(
#    data = 'data/RadioGalaxyNetSeg.yaml',
#    epochs = 300,
#    imgsz = 450,
#    device = 0,
#    name = 'yolov9c-seg_GAL1')

#results.show()



#TESTING
reset_seed(111)

model = YOLO('runs/segment/yolov9c-seg_GAL1/weights/best.pt')
results = model.val(task = 'segment', data = 'data/RadioGalaxyNetSeg.yaml', split = 'test', imgsz = 450, plots = True, save_json = True)
print("map50-95")
print(results.seg.map)    # map50-95
print("map50")
print(results.seg.map50)  # map50
print("map75")
print(results.seg.map75)  # map75
print("map50-95 - each category")
print(results.seg.maps)   # a list contains map50-95 of each category

print(results.seg)


print("IOU values:")

# Retrieve segmentation IOU values for each class

from ultralytics import YOLO

# Access IoU metrics
iou_score = results.metrics['mask'].iou  # Replace 'box' with 'mask' for segmentation
print(f"IoU score: {iou_score}")


