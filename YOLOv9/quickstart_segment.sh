#!/bin/bash

#SBATCH --job-name=yolo_zach_segmentation
#SBATCH --account=ml20

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

#SBATCH --mem-per-cpu=60000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=6-00:00:00

# Set the file for output (stdout)
#SBATCH --output=/home/zacharid/ml20/FYP_GALAXY/YOLOv9/segmentation.out

# Set the file for error log (stderr)
#SBATCH --error=/home/zacharid/ml20/FYP_GALAXY/YOLOv9/segmentation.err

# Command to run a gpu job
# For example:
source ~/ml20_scratch/miniconda3/bin/activate
conda activate yolo
cd ~/ml20/FYP_GALAXY/YOLOv9/yolov9-main

python zach_yolo_segment_v1.py