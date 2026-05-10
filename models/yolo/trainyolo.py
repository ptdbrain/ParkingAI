from ultralytics import YOLO

model = YOLO("yolo26n.pt") # Build a new model from 'yolo26n.yaml' configuration

data_yolo = 'D:\\Project\\edge_parking_system\\dataset\\processed\\detection\\data.yaml'

results = model.train(data=data_yolo, epochs=100, imgsz=640, cache = True)