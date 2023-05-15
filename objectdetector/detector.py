import multiprocessing as mp
import queue
import time

import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from visionapi.detector_pb2 import DetectionOutput
from visionapi.videosource_pb2 import VideoFrame

from .config import ObjectDetectorConfig
from .errors import *


class Detector:
    def __init__(self, config: ObjectDetectorConfig) -> None:
        self.config = config
        self.stop_event = mp.Event()
        self.input_queue = mp.Queue(5)
        self.output_queue = mp.Queue(5)
        self.detector_loop = _DetectorLoop(config, self.stop_event, self.input_queue, self.output_queue)

    def start(self):
        self.detector_loop.start()

    def stop(self):
        self.stop_event.set()

    def put_frame(self, frame, block=True, timeout=10):
        self._assert_running()

        try:
            self.input_queue.put(frame, block, timeout)
        except queue.Full:
            raise InputFullError(f'No frame could be added to the input queue after having waited {timeout}s')

    def get_detection(self, block=True, timeout=10):
        self._assert_running()

        try:
            return self.output_queue.get(block, timeout)
        except queue.Empty:
            raise NoDetectionError(f'No detection has been received after having waited {timeout}s')

    def _assert_running(self):
        if self.stop_event.is_set():
            raise StoppedError('Detector has already been stopped')
        

class _DetectorLoop(mp.Process):
    def __init__(self, config: ObjectDetectorConfig, stop_event, input_queue, output_queue, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.stop_event = stop_event
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.model = None
        self.device = None
        self.input_image_size = None

    @torch.no_grad()
    def run(self):
        self._setup_model()

        while not self.stop_event.is_set():
            try: 
                input_image, frame_proto = self._get_next_image(block=False)
            except queue.Empty:
                time.sleep(0.01)
                continue
                
            inf_image = self._prepare_input(input_image)

            yolo_prediction = self.model(inf_image)
            predictions = non_max_suppression(yolo_prediction)[0]

            predictions[:, :4] = scale_boxes(inf_image.shape[2:], predictions[:, :4], input_image.shape[:2]).round()

            try:
                self.output_queue.put(self._create_output(predictions, frame_proto), block=False)
            except queue.Full:
                time.sleep(0.01)
            
        self.input_queue.cancel_join_thread()
        self.output_queue.cancel_join_thread()

    def _setup_model(self):
        self.device = torch.device(self.config.model_config.device)
        self.model = AutoBackend(
            self._yolo_weights(), 
            device=self.device
        )
        self.input_image_size = check_imgsz(self.config.inference_size, stride=self.model.stride)

    def _yolo_weights(self):
        return f'yolov8{self.config.model_config.size.value}.pt'
    
    def _get_next_image(self, block=True):
        frame_proto_raw = self.input_queue.get(block)

        frame_proto = VideoFrame()
        frame_proto.ParseFromString(frame_proto_raw)

        input_image = np.frombuffer(frame_proto.frame_data, dtype=np.uint8).reshape(frame_proto.shape)
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

