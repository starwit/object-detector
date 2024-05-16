from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Export the model to TensorRT format
model.export(
    format='engine', 
    device=0, 
    imgsz=(1280,1280), 
    dynamic=True, 
    batch=1,
    half=True,
)  # creates 'yolov8m.engine'