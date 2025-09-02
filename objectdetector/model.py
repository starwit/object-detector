from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import logging
from numpy.typing import NDArray
from prometheus_client import Summary
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import non_max_suppression

from .config import ModelConfig, LogLevel

logger = logging.getLogger(__name__)

MODEL_DURATION = Summary('object_detector_model_duration', 'How long the model call takes (without NMS)')
NMS_DURATION = Summary('object_detector_nms_duration', 'How long non-max suppression takes')

# Source: https://github.com/ultralytics/ultralytics/blob/248405ab9221a56b2a13d28d7a56db4d9fdba2ae/ultralytics/nn/autobackend.py#L160
index_to_type = {
    0: 'pt',
    1: 'jit',
    2: 'onnx',
    3: 'xml',
    4: 'engine',
    5: 'coreml',
    6: 'saved_model',
    7: 'pb',
    8: 'tflite',
    9: 'edgetpu',
    10: 'tfjs',
    11: 'paddle',
    12: 'mnn',
    13: 'ncnn',
    14: 'imx',
    15: 'rknn',
    16: 'triton',
}

class ModelType(str, Enum):
    PT = 'PT'
    OPENVINO = 'OPENVINO'
    TENSORRT = 'TENSORRT'

class Model:
    def __init__(self, config: ModelConfig, log_level: LogLevel = LogLevel.INFO):
        logger.setLevel(log_level.value)
        self._config = config
        self._model_type = self._determine_model_type(config.weights_path)
        logger.info(f'Detected model type: {self._model_type.name}')
        self._check_weights_path(config.weights_path)
        self._check_configuration()
        self._model = AutoBackend(str(config.weights_path), self._get_device(config.device), config.fp16)

    def _determine_model_type(self, weights_path: Path) -> ModelType:
        model_type = AutoBackend._model_type(str(weights_path))
        if model_type[0]:
            return ModelType.PT
        elif model_type[3]:
            return ModelType.OPENVINO
        elif model_type[4]:
            return ModelType.TENSORRT
        else:
            raise ValueError(f'Unsupported model type found at {weights_path}: {index_to_type[model_type.index(True)]}')
        
    def _check_weights_path(self, weights_path: Path):
        if not weights_path.exists() and not self._config.auto_download:
            raise ValueError(f'No such file or directory found at {weights_path} and auto_download==False')
        
    def _check_configuration(self):
        if self._config.device.startswith('intel:') and self._model_type != ModelType.OPENVINO:
            raise ValueError(f'Only OpenVINO models are supported with `intel:*` device type')
        
    def _get_device(self, device_str: str) -> torch.device | str:
        if self._model_type == ModelType.TENSORRT:
            return torch.device(device_str)
        return device_str

    def __call__(self, batch: NDArray) -> List[Results]:
        input_tensor = self._create_input_tensor(batch)
        
        with MODEL_DURATION.time():
            yolo_prediction = self._model(input_tensor)

        with NMS_DURATION.time():
            predictions = non_max_suppression(
                yolo_prediction, 
                conf_thres=self._config.confidence_threshold, 
                iou_thres=self._config.iou_threshold,
                classes=self._config.classes,
                agnostic=self._config.nms_agnostic,
            )

        return predictions
    
    def _create_input_tensor(self, batch: NDArray) -> torch.Tensor:
        numpy_batch_ct = np.ascontiguousarray(batch)
        batch_tensor = torch.from_numpy(numpy_batch_ct).float() / 255.0

        # Currently, we cannot use torch.device('xpu') with Intel GPU. As soon as we've managed to enable xpu support for pytorch we can remove the condition
        # We currently rely on Ultralytics intel support to move the tensor to the GPU internally
        if self._model_type == ModelType.TENSORRT or self._config.device == 'cuda':
            batch_tensor = batch_tensor.to(torch.device(self._config.device))
            
        return batch_tensor
    
    @property
    def stride(self) -> int:
        return self._model.stride
    
    @property
    def names(self) -> Dict[int, str]:
        return self._model.names