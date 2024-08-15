from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', workers = 0, device = 'mps', batch=16, epochs=16, imgsz=640)