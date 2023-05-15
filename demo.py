from objectdetector.detector import Detector
from objectdetector.config import ObjectDetectorConfig, ModelSizeEnum, YoloV8Config
import time 
from visionapi.videosource_pb2 import VideoFrame
from visionapi.detector_pb2 import DetectionOutput
from google.protobuf.json_format import MessageToDict
import json
import cv2
import numpy as np

frame = cv2.imread('./test.jpg')

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

def frame_from_proto(frame):
    return np.frombuffer(frame.frame_data, dtype=np.uint8).reshape(frame.shape)

proto_frame = to_proto(frame)

detector = Detector(
    ObjectDetectorConfig(
        model_config=YoloV8Config(size=ModelSizeEnum.NANO, device='cuda:0'),
        inference_size=(640,640)
    )
)

detector.start()

detector.put_frame(proto_frame)
det_output = deserialize_proto(detector.get_detection())

detector.stop()

print(json.dumps(MessageToDict(det_output)['detections'], indent=2))
cv2.imshow('win', frame_from_proto(det_output.frame))
cv2.waitKey(0)