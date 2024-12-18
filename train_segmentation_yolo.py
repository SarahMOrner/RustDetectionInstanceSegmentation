from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO('yolov8n-seg.pt')

model.train(data='data.yaml', epochs=50, imgsz=640, task='segment')
