import os
import time

import cv2
from google.protobuf.text_format import MessageToString
from visionapi.detector_pb2 import DetectionOutput
from visionapi.videosource_pb2 import VideoFrame

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
    vf.shape[:] = frame.shape
    vf.frame_data = frame.tobytes()

    return vf.SerializeToString()

def deserialize_proto(message):
    det_output = DetectionOutput()
    det_output.ParseFromString(message)
    return det_output

def write_detection(basename, path, detection):
    with open(os.path.join(path, basename) + '.pbtxt', 'w') as f:
        f.write(MessageToString(detection))


detector = Detector(
    ObjectDetectorConfig(
        model_config=YoloV8Config(size=ModelSizeEnum.NANO, device='cuda:0'),
        inference_size=(640,640)
    )
)

detector.start()

for basename, frame in frame_iter('.demo_frames'):
    detector.put_frame(to_proto(frame))
    detection = deserialize_proto(detector.get_detection())
    write_detection(basename, '.demo_detections', detection)

detector.stop()
