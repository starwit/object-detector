import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.checks import check_imgsz

from .config import ObjectDetectorConfig


class Detector:
    def __init__(self, config: ObjectDetectorConfig) -> None:
        self.config = config
        self.model = None
        self.device = None
        self.input_image_size = None

    @torch.no_grad()
    def run(self):
        self._setup_model()

        while True:
            input_img = self._prepare_input(image)
            prediction = self.model(input_img)
            

    def _setup_model(self):
        self.device = torch.device(self.config.model_config.device)
        self.model = AutoBackend(
            self._yolo_weights(), 
            device=self.device
        )
        self.input_image_size = check_imgsz(self.config.inference_size, stride=self.model.stride)

    def _yolo_weights(self):
        return f'yolov8{self.config.model_config.size.value}.pt'
    
    def _prepare_input(self, image):
        out_img = LetterBox(self.config.inference_size, auto=True, stride=self.model.stride)(image=image)
        out_img = out_img.transpose((2, 0, 1))[::-1]
        out_img = np.ascontiguousarray(out_img)
        out_img = torch.from_numpy(out_img).to(self.device).float() / 255.0
        # TODO Expand for batch dim?

