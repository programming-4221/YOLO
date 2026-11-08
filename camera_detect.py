from ultralytics import YOLO

model = YOLO("yolov8n.pt")

for result in model.predict(
        source = 0,
        show = True,
        stream = True,
        conf = 0.2,
        imgsz = 416,
        device = "cpu"
    ):
        pass