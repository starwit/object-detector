# YOLOv8 TensorRT optimization
## Working configs
The following parameters work reasonably well (with only some optimization routines skipped due to limited memory) on an A2000 6GB
and bring a considerable performance uplift (roughly x3):
- imgsz: 1280
- dynamic: True
- batch: 1
- half: True
For inferencing it would probably be ideal to optimize for batch size 2 on an A2000 12GB,
as the optimization process appears to consume significantly more memory than inferencing.