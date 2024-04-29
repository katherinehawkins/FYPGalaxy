import os
import argparse
import json
import sys
from pathlib import Path
import random
import numpy as np
import torch
from tqdm import tqdm
import subprocess
from shapely.geometry import Polygon
import cv2
from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.detection import MeanAveragePrecision

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)



print("ATTEMPTING TO INSTALL REQUIREMENTS:")
#os.system("pip install -r requirements.txt")
#os.system("pip install ultralytics --upgrade")
#os.system("pip install torchmetrics")



from ultralytics import YOLO

#TRAINING YOLOv9
# Initialize the YOLO model
model = YOLO('runs/segment/tune/weights/best.pt', task = 'segment')


model.tune(data='data/RadioGalaxyNetSeg.yaml', epochs=200, iterations=30, optimizer='SGD', plots=True, save=True, val=True, batch = 8, patience = 300)



"""
results = model.train(
    data = 'data/RadioGalaxyNetSeg.yaml',
    epochs = 300,
    imgsz = 450,
    device = 0,
    name = 'yolov9c-seg_GAL1')
"""
#results.show()


"""
#TESTING
reset_seed(111)

#model = YOLO('runs/segment/yolov9c-seg_GAL1/weights/best.pt')
results = model.val(task = 'segment', data = 'data/RadioGalaxyNetSeg.yaml', split = 'test', imgsz = 450, plots = True, save_json = True)
print("map50-95")
print(results.seg.map)    # map50-95
print("map50")
print(results.seg.map50)  # map50
print("map75")
print(results.seg.map75)  # map75
print("map50-95 - each category")
print(results.seg.maps)   # a list contains map50-95 of each category




#TESTING

def solve(path_to_weight_pt_file, images_folder, labels_folder):
    # Loading the trained segmentation model
    model = YOLO(path_to_weight_pt_file)

    # Initialize metrics
    metrics = MetricCollection([Accuracy(task='multiclass', num_classes=5, average = 'macro'), 
                                JaccardIndex(task='multiclass', num_classes=5)])

    total_loss = 0
    num_samples = 0

    for img_file in os.listdir(images_folder):
        if img_file.endswith(".png"):
            image_path = os.path.join(images_folder, img_file)
            label_file = os.path.join(labels_folder, img_file.split('.')[0] + '.txt')

            # Read the image
            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            # Read label file
            with open(label_file, "r") as f:
                for line in f:
                    class_id, *poly = line.strip().split()
                    class_id = int(class_id)
                    poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
                    poly *= [w, h]  # Unscale
                    poly = poly.astype(int)  # Convert to int

                    # Now compute the predicted mask
                    res = model.predict(image_path)
                    for r in res:
                        if r.masks is not None:  # Check if masks are present
                            for pred_vector in r.masks.xy:
                                pred_vector = np.array(pred_vector, dtype=int)  # Convert to int

                                # Update metrics
                                logits = torch.zeros((1, 4, h, w))  # Assuming 4 classes
                                logits[0, class_id, pred_vector[:, 1], pred_vector[:, 0]] = 1
                                true_mask = torch.zeros((1, 4, h, w))
                                true_mask[0, class_id, poly[:, 1], poly[:, 0]] = 1
                                metrics.update(logits, true_mask)

    # Compute metrics
    scores = metrics.compute()
    print("Accuracy:", scores["MulticlassAccuracy"])
    print("Jaccard Index:", scores["MulticlassJaccardIndex"])


# Example usage
solve("runs/segment/tune/weights/best.pt", "../datasets/RadioGalaxyNET/data/RadioGalaxyNETSeg/images/val", "../datasets/RadioGalaxyNET/data/RadioGalaxyNETSeg/labels/val")


#metrics.update(logit.detach().cpu(), true_mask.cpu())
#scores = metrics.compute()
#metrics = MetricCollection([Accuracy(task='multiclass', num_classes=4), JaccardIndex(task='multiclass', num_classes=4)])


#metrics.reset()

"""
