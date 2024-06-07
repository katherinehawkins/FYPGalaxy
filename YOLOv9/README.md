<h1>YOLOv9</h1>

There are two versions of YOLOv9, the official one from Ultralytics and the original one from Wang et al. We implement both.

In order to set up YOLOv9, the entire folder needs to be downloaded from this link:
https://drive.google.com/file/d/1XsMn4PrEYPve8RjJ7-QWamfbjT6ayTeH/view?usp=drive_link

<h2>Detection</h2>

**To TRAIN an _official_ detection model, you need to access the file yolov9-main/zach_yolo_v1.py**
- Uncomment lines 20-23 so that the requirements are installed
- Lines 27 & 28 decide whether you train a model from scratch or a pretrained one, respectively. Comment out whichever you do not wish to use.
- The next section outlines two options for training, the first being a simplified train, the second being a more complex one. Once again, comment out whichever you do not wish to use
- Uncomment Line 91 to show the training results.
- The trained model will be saved in yolov9-main/runs/detect
- The testing section of the code will test your model for you and save the results into yolov9-main/runs/detect
- Run code
<br>
  
**To FINE TUNE an _official_ detection model, you need to access the file yolov9-main/zach_yolo_v1.py**
- Leave the code as it is upon download i.e. with the training lines commented out
- Uncomment lines 20-23 so that the requirements are installed
- Line 28 will specify which model you wish to fine tune
- Uncomment Line 95 and specify your training parameters
- Comment out the testing section
- The fine tuning data will be saved in yolov9-main/runs/detect
- Run code
<br>

**To TRAIN an _unofficial_ detection model, you need to access the file yolov9-main/zach_yolo_v1.py**
- Uncomment lines 20-23 so that the requirements are installed
- Comment out Line 28
- Comment out the testing section
- Uncomment Lines 167 & 168 and specify your training parameters
- More hyperparameters can be adjusted in yolov9-main/data/hyps/hyp.tuned_params_pretrained_it119.yaml
- Feel free to create a new yaml file, just update the reference in Line 168
- The trained model will be saved in yolov9-main/runs/train
- To test the model, uncomment Lines 173 & 174 and update the model reference
- The testing data will be saved in yolov9-main/runs/test
- Run code
<br>

**These files can be run on MASSIVE or a different terminal if needed**
- If running on MASSIVE, update the file quickstart.sh with your requirements and run the command 'sbatch quickstart.sh'
- If running on a different terminal, the zach_yolo_v1.py file can be run directly


<br>

<h2>Segmentation</h2>

**To TRAIN a segmentation model, you need to access the file yolov9-main/zach_yolo_segment_v1.py**
- Uncomment lines 24-26 so that the requirements are installed
- Line 35 initialises the model. Update the reference to the yolov9-main/models/segment/yolov9c-seg.yaml file if you wish to train from scratch
- The training section on Line 45 outlines a basic train. More hyperparameters can be added if desired.
- The trained model will be saved in yolov9-main/runs/segment
- The testing section of the code will test your model for you and save the results into yolov9-main/runs/segment
- Update Line 122 to compute the IOU for your segmentation model and place it in files segmentation.err and segmentation.out
- Run code
<br>
  
**To FINE TUNE a segmentation model, you need to access the file yolov9-main/zach_yolo_segment_v1.py**
- Uncomment lines 20-23 so that the requirements are installed
- Line 35 will specify which model you wish to fine tune
- Uncomment Line 39 and specify your training parameters
- Comment out the testing section
- Comment out Line 122
- The fine tuning data will be saved in yolov9-main/runs/segment
- Run code

<br>

**Once again, these files can be run on MASSIVE or a different terminal if needed**
- If running on MASSIVE, update the file quickstart_segment.sh with your requirements and run the command 'sbatch quickstart.sh'
- If running on a different terminal, the zach_yolo_segment_v1.py file can be run directly

<br>

<h2>Happy training!</h2>
