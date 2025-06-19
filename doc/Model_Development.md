# Nvidia TensorRT
A model optimized with Nvidia TensorRT can be much faster (200-300% increase in throughput), but is a lot more difficult to handle
compare to the pytorch model, which is very portable.\
Other detector settings strongly depend on the parameters that were used for the TensorRT optimizization, e.g. inference size and batch size.\
Also, the installed Nvidia driver and CUDA component versions need to be tightly managed.

## Running TensorRT optimizations
Nvidia documentation states that ideally the TensorRT optimizations should be run on exactly the same GPU model that will be used for inferencing. If that is not possible, the next best thing is running the optimization process on the smallest GPU variant (that is going to be used) of a certain generation and it should translate reasonably well to bigger variants.

The following parameters work reasonably well with the m-sized YOLOv8 model (with only some optimization routines skipped due to limited memory) on an A2000 6GB
and bring a considerable performance uplift (roughly 3x):
- imgsz: 1280
- dynamic: True
- batch: 1
- half: True
For inferencing it would probably be ideal to optimize for batch size 2 on an A2000 12GB,
as the optimization process appears to consume significantly more memory than inferencing.

### Example using ultralytics TensorRT adapter
```python
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
```

### Example using TensorRT tools on Jetson Orin NX (temporary notes)
Additionally to running the Python snippet from above (which has not been tested on the Jetson, but should work!),
it is also possible to use the TensorRT tools to do the same optimization.\
Assuming that the TensorRT tools are installed, you can use the commandline tool `trtexec` to convert a model. First, you have to convert the model to ONNX format.
```bash
# Export YOLO model to ONNX format (apparently this defines the maximum input dimensions and batch size the tensorrt model will be able to accept in the next step, even if the latter is also set to dynamic)
yolo export model=yolov8m.pt format=onnx simplify=True dynamic=True batch=4 imgsz=1280 half=True

# Convert ONNX model to TensorRT engine format, explicitly specifying the range of dynamic dimensions
trtexec --onnx=yolov8m.onnx --saveEngine=yolov8m.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x736x1280 --maxShapes=images:4x3x736x1280
```