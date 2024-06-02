# Faster RCNN and DINO

## Step 1: Download the required packages

**Python requirements:** Python 3.12.2

```
pip install requirements.txt
```

## Step 2: Generate the Augmented Datasets
1. To remove the first infered channel from the images in the dataset:
```
python DINO/data_augmentation rgn_aug.py
```
2. To augment the MiraBest dataset:
```
python DINO/data_augmentation image_generator.py
```

## Step 3: Pre-training
The pretraining scripts for DINO is recorded in the 'DINO\DINO\GalDINO.ipynb folder.

## Step 4: Faster RCNN
The pretraining scripts for DINO is recorded in the 'FasterRCNN\fasterRCNN.ipynb folder.