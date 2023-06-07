import logging
import time
from typing import Any

import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from visionapi.messages_pb2 import DetectionOutput, Metrics, VideoFrame

from .config import ObjectDetectorConfig

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, config: ObjectDetectorConfig) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)

        self.model = None
        self.device = None
        self.input_image_size = None

        self._setup_model()

    def __call__(self, input_proto, *args, **kwargs) -> Any:
        return self.get(input_proto)

    @torch.no_grad()
    def get(self, input_proto):
            
        input_image, frame_proto = self._unpack_proto(input_proto)
            
        inference_start = time.monotonic_ns()
        inf_image = self._prepare_input(input_image)

        yolo_prediction = self.model(inf_image)

        predictions = non_max_suppression(
            yolo_prediction, 
            conf_thres=self.config.model_config.confidence_threshold, 
            iou_thres=self.config.model_config.iou_threshold
        )[0]
        predictions[:, :4] = scale_boxes(inf_image.shape[2:], predictions[:, :4], input_image.shape[:2]).round()

        inference_time_us = (time.monotonic_ns() - inference_start) // 1000
        return self._create_output(predictions, frame_proto, inference_time_us)

    def _setup_model(self):
        logger.info('Setting up object-detector model...')
        self.device = torch.device(self.config.model_config.device)
        self.model = AutoBackend(
            self._yolo_weights(),
            device=self.device,
            fp16=self.config.model_config.fp16_quantization
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

    def _create_output(self, predictions, frame_proto, inference_time_us):
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

        output.metrics.detection_inference_time_us = inference_time_us

        return output.SerializeToString()
