import os
import time

import cv2
from google.protobuf.text_format import MessageToString
from visionapi.messages_pb2 import DetectionOutput
from visionapi.messages_pb2 import VideoFrame, Shape

from objectdetector.config import (ModelSizeEnum, ObjectDetectorConfig,
                                   YoloV8Config)
from objectdetector.detector import Detector


def frame_iter(path):
    files = sorted(os.listdir(path))
    for file in files:
        basename = os.path.splitext(file)[0]
        cv2_frame = cv2.imread(os.path.join(path, file))
        yield basename, cv2_frame

def to_proto(frame):
    vf = VideoFrame()
    vf.timestamp_utc_ms = time.time_ns() // 1000

    shape = Shape()
    shape.height, shape.width, shape.channels = frame.shape[0], frame.shape[1], frame.shape[2]
    vf.shape.CopyFrom(shape)
    vf.frame_data = frame.tobytes()

    return vf.SerializeToString()

def deserialize_proto(message):
    det_output = DetectionOutput()
    det_output.ParseFromString(message)
    return det_output

def write_detection_text(basename, path, detection: DetectionOutput):
    with open(os.path.join(path, basename) + '.pbtxt', 'w') as f:
        f.write(MessageToString(detection))

def write_detection_bin(basename, path, detection: DetectionOutput):
    with open(os.path.join(path, basename) + '.bin', 'wb') as f:
        f.write(detection.SerializeToString())


detector = Detector(
    ObjectDetectorConfig(
        model_config=YoloV8Config(size=ModelSizeEnum.NANO, device='cuda:0'),
        inference_size=(640,640)
    )
)

for basename, frame in frame_iter('.demo_frames'):
    detection = deserialize_proto(detector.get(to_proto(frame)))
    print(f'Inference time: {detection.metrics.detection_inference_time_us} us')
    # write_detection_text(basename, '.demo_detections', detection)
    write_detection_bin(basename, '.demo_detections', detection)

