# Object Detector

This component is part of the Starwit Awareness Engine (SAE). See umbrella repo here: https://github.com/starwit/starwit-awareness-engine

## How to set up
- Make sure you have Poetry installed (otherwise see here: https://python-poetry.org/docs/#installing-with-the-official-installer)
- Set environment variable `NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True` (otherwise `tensorrt-*` installation will fail)
- Run `poetry install`

# Nvidia TensorRT
A model optimized with Nvidia TensorRT can be much faster (200-300% increase in throughput), but is a lot more difficult to handle
compare to the pytorch model, which is very portable.\
Only set `use_tensorrt` in the model config to `True` if you know what you are doing!\
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

### Example
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
