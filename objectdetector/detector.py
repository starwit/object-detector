from typing import Any
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from visionapi.detector_pb2 import DetectionOutput
from visionapi.videosource_pb2 import VideoFrame

from .config import ObjectDetectorConfig


class Detector:
    def __init__(self, config: ObjectDetectorConfig) -> None:
        self.config = config

        self.model = None
        self.device = None
        self.input_image_size = None

        self._setup_model()

    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)

    @torch.no_grad()
    def get(self, input_proto):
            
        input_image, frame_proto = self._unpack_proto(input_proto)
            
        inf_image = self._prepare_input(input_image)

        yolo_prediction = self.model(inf_image)
        predictions = non_max_suppression(yolo_prediction)[0]

        predictions[:, :4] = scale_boxes(inf_image.shape[2:], predictions[:, :4], input_image.shape[:2]).round()

        return self._create_output(predictions, frame_proto)

    def _setup_model(self):
        self.device = torch.device(self.config.model_config.device)
        self.model = AutoBackend(
            self._yolo_weights(), 
            device=self.device
        )
        self.input_image_size = check_imgsz(self.config.inference_size, stride=self.model.stride)

    def _yolo_weights(self):
        return f'yolov8{self.config.model_config.size.value}.pt'
    
    def _unpack_proto(self, proto_bytes):
        frame_proto = VideoFrame()
        frame_proto.ParseFromString(proto_bytes)

        input_image = np.frombuffer(frame_proto.frame_data, dtype=np.uint8) \
            .reshape((frame_proto.shape.height, frame_proto.shape.width, frame_proto.shape.channels))
        return input_image, frame_proto
    
    def _prepare_input(self, image):
        out_img = LetterBox(self.config.inference_size, auto=True, stride=self.model.stride)(image=image)
        out_img = out_img.transpose((2, 0, 1))[::-1]
        out_img = np.ascontiguousarray(out_img)
        out_img = torch.from_numpy(out_img).to(self.device).float() / 255.0
        return out_img.unsqueeze(0)

    def _create_output(self, predictions, frame_proto):
        output = DetectionOutput()

        for pred in predictions:
            detection = output.detections.add()

            detection.bounding_box.min_x = int(pred[0])
            detection.bounding_box.min_y = int(pred[1])
            detection.bounding_box.max_x = int(pred[2])
            detection.bounding_box.max_y = int(pred[3])

            detection.confidence = float(pred[4])
            detection.class_id = int(pred[5])

        output.frame.CopyFrom(frame_proto)

        return output.SerializeToString()
