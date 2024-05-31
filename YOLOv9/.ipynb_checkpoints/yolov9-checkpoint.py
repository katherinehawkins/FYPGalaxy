from ultralytics import YOLO

# Build a YOLOv9c model from scratch
model = YOLO('yolov9c.yaml')

# Build a YOLOv9c model from pretrained weight
model = YOLO('yolov9c_local.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='RadioGalaxyNet.yaml', epochs=100, imgsz=450)


# Run inference with the YOLOv9c model on the 'bus.jpg' image
results = model('path/to/bus.jpg')

print(results)

model.save('saved_model_v1')




#extras
#to train?
#python train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15