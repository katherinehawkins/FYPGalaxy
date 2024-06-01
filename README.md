#  Unlocking Celestial Enigma With AI

**Supervisor:** Mehrtash T. Harandi

**Authors:** Katherine Hawkins, Muhammad Suleman, Zach Drinkall, Yide Tao

**Resources:**
- [Full Report](https://github.com/katherinehawkins/FYPGalaxy/Final%20Report%20-%20ENG4072.pdf)
- [Video Presentation](https://monash.au.panopto.com/Panopto/Pages/Viewer.aspx?id=0a280098-8c8e-4223-92a4-b17300e56a46)

## Project Discription

The project aims to showcase different tools for the State of the Art detection and segmentation of Radio Galaxies, exploring the effects of different state of the art vision model architectures such as YOLO, SAM and DINO + Faster RCNN. The [dataset](https://data.csiro.au/collection/csiro%3A61068v1) used in this project is from the [RadioGalaxyNET](https://arxiv.org/abs/2312.00306) provided by CSIRO.

<p float="middle">
  <img src="0_images/demo1_seg_detect.png?raw" alt="Segmentation Detection Demo"/>
  <p>Figure 1: Example of segmentation + detection Using YOLO v9 Model.</p>
</p>

**Model Explored**
1. [YOLO V9](https://github.com/katherinehawkins/FYPGalaxy/YOLOv9)
2. [Segment Anything Model](https://github.com/katherinehawkins/FYPGalaxy/SAM)
3. [DINO + Faster RCNN](https://github.com/katherinehawkins/FYPGalaxy/DINOFasterRCNN)

## Results
**Detection Results**
| Models                                   | mAP 50 | Status |
|------------------------------------------|--------|--------|
| YOLOv9 (Wang et al. Version)             | 0.817  |Our Result|
| YOLOv9 (Ultralytics Version)             | 0.798  |Our Result|
| Faster RCNN (DINO + DARGN)               | 0.628  |Our Result|
| Faster RCNN (Baseline)                   | 0.588  |Our Result|
| [Gal-DINO (DETR with Improved deNoising anchOr boxes)](https://arxiv.org/abs/2312.00306) | 0.602  |Previous Best Model|

**Segmentation Results**
| Models                                | Type     | mIoU  | Status |
|---------------------------------------|----------|-------|--------|
| YOLOv9                                 | Panoptic | 0.508 |Our Result|
| Faster-RCNN (DINO + DARGN) + segmentation head | Panoptic | 0.684 |Our Result|
| Faster-RCNN (Baseline) + segmentation head | Panoptic | 0.655 |Our Result|
| SAM with LoRA                          | Semantic | 0.451 |Our Result|
| U-Net                                  | Semantic | 0.595 |Our Result|

## Installation
1. Download the repository using:
```
git clone https://github.com/facebookresearch/segment-anything?tab=readme-ov-file
```

2. For more detailed instruction in how to use each of the models please refer to the readme page in each corresponding subfolder.

## Model Checkpoints
| Models                                   | Links | 
|------------------------------------------|--------|
| Yolo v9            | [checkpoint]()|
| DINO Weights          | [checkpoint]()|
| DINO + Faster RCNN Weights         | [checkpoint]()|

## License
The model is licensed under the Apache 2.0 license.