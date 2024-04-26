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
model = YOLO('runs/detect/yolov9-c-GAL7_tuned_pretw_it166_newyaml/weights/best.pt')
results = model.train(
    data = 'data/RadioGalaxyNet.yaml',
    epochs = 500,
    patience = 500,
    batch = 8,
    imgsz = 450,
    workers = 8,
    device = 0,
    name = 'yolov9-c-GAL8_tuned_and_retrained',
    optimizer = 'SGD',
    pretrained = True,
    close_mosaic = 10,
    lr0 = 0.00838,
    lrf = 0.00971,
    momentum = 0.98,
    weight_decay = 0.00051,
    warmup_epochs = 3.30461,
    warmup_momentum = 0.95,
    warmup_bias_lr = 0.1,
    box = 7.64199,
    cls = 0.50743,
    dfl = 1.94592,
    hsv_h = 0.01203,
    hsv_s = 0.66025,
    hsv_v = 0.54135,
    degrees = 0.0,
    translate = 0.09604,
    scale = 0.63533,
    shear = 0.0,
    perspective = 0.0,
    flipud = 0.0,
    fliplr = 0.37607,
    mosaic = 1.0,
    mixup = 0.0,
    copy_paste = 0.0,
    mask_ratio = 4,
    pose = 12,
    erasing  = 0.4,
    crop_fraction = 1.0)




    

results.show()


#model.tune(data='data/RadioGalaxyNet.yaml', epochs=300, iterations=20, optimizer='SGD', plots=True, save=False, val=True, batch = 8, patience = 300)




#TESTING
#reset_seed(111)
#metrics = model.val(data = 'data/RadioGalaxyNet.yaml', split = 'test', imgsz = 450, plots = True, save_json = True)
#print("map50-95")
#print(metrics.box.map)    # map50-95
#print("map50")
#print(metrics.box.map50)  # map50
#print("map75")
#print(metrics.box.map75)  # map75
#print("map50-95 - each category")
#print(metrics.box.maps)   # a list contains map50-95 of each category




#TRAINING YOLOV8
#model = YOLO('models/detect/yolov8.yaml', task='detect')
#results = model.train(
#    data = 'data/RadioGalaxyNet.yaml',
#    name = 'yolov8-GAL1_test',
#    epochs = 50)


#TRAINING YOLOv9
# Initialize the YOLO model
#model = YOLO('models/detect/yolov9c.yaml', task='detect')
#results = model.train(
#    data = 'data/RadioGalaxyNet.yaml',
#    epochs = 1250,
#    patience = 200,
#    batch = 8,
#    imgsz = 450,
#    workers = 8,
#    device = 0,
#    name = 'yolov9-c-GAL7_tuned_pretw_it166_newyaml',
#    optimizer = 'SGD',
#    close_mosaic = 15,
#    lr0 = 0.00819,
#    lrf = 0.01594,
#    momentum = 0.88838,
#    weight_decay = 0.00038,
#    warmup_epochs = 3.20084,
#    warmup_momentum = 0.95,
#    warmup_bias_lr = 0.1,
#    box = 5.72279,
#    cls = 0.47753,
#    dfl = 1.45705,
#    hsv_h = 0.01404,
#    hsv_s = 0.62532,
#    hsv_v = 0.29034,
#    degrees = 0.0,
#    translate = 0.09632,
#    scale = 0.3307,
#    shear = 0.0,
#    perspective = 0.0,
#    flipud = 0.0,
#    fliplr = 0.26038,
#    mosaic = 0.89249,
#    mixup = 0.0,
#    copy_paste = 0.0)

#results.show()


# Tune hyperparameters on COCO8 for 30 epochs
#model.tune(data='data/RadioGalaxyNet.yaml', epochs=100, iterations=100, optimizer='SGD', plots=True, save=False, val=True)


#print("TRAINING ATTEMPTING TO BEGIN:")
#os.system("python train_dual.py --workers 8 --device 0 --batch 8 --data data/RadioGalaxyNet.yaml --img 450 --optimizer SGD --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c-GAL7_tuned_pretw_it119_2 --hyp hyp.tuned_params_pretrained_it119.yaml --min-items 0 --epochs 1250 --close-mosaic 15 --patience 200")




#GAL1
#batch 8>4

#GAL3
#batch 4>8
#ADAM lre-3

#GAL4
#SGD
#degrees = 90
#flip up down = 0.5




#to change:
# - optimiser
# - learning rate
# - data augmentation
# - image rotation, scaling, etc.




#print("TESTING ALL:")
#os.system("python test_dual.py --data data/RadioGalaxyNet.yaml --img 450 --batch 1 --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/yolov9-c-GAL6_tuned_pretw_it119/weights/best.pt' --save-json --name yolov9-c-GAL1_test --task 'test'")

#os.system("python test_dual.py --data data/RadioGalaxyNet.yaml --img 450 --batch 1 --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/yolov9-c-GAL2_batch4_pat200/weights/best.pt' --save-json --name yolov9-c-GAL2_test --task 'test'")

#os.system("python test_dual.py --data data/RadioGalaxyNet.yaml --img 450 --batch 1 --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/yolov9-c-GAL3_batch8_pat200_ADAM_lr1e-3/weights/best.pt' --save-json --name yolov9-c-GAL3_test --task 'test'")

#os.system("python test_dual.py --data data/RadioGalaxyNet.yaml --img 450 --batch 1 --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/yolov9-c-GAL4_RF_batch8_pat200/weights/best.pt' --save-json --name yolov9-c-GAL4_test --task 'test'")

#os.system("python test_dual.py --data data/RadioGalaxyNet.yaml --img 450 --batch 1 --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/yolov9-c-GAL5_tuned_pretw_it6/weights/best.pt' --save-json --name yolov9-c-GAL5_test --task 'test'")


#TO DO:
# - Try to force update on ultralytics
# - Retrain and try the tuning again
# - Try to run the tune on it
# - If that doesn't work, keep messing with hyperparams

