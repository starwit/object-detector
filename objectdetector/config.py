from pydantic import BaseModel
from enum import Enum

class LogLevel(str, Enum):
    CRITICAL = 'CRITICAL'
    ERROR = 'ERROR'
    WARNING = 'WARNING'
    INFO = 'INFO'
    DEBUG = 'DEBUG'

class ModelSizeEnum(str, Enum):
    NANO = 'n'
    SMALL = 's'
    MEDIUM = 'm'
    LARGE = 'l'
    XLARGE = 'x'

class YoloV8Config(BaseModel):
    size: ModelSizeEnum = ModelSizeEnum.NANO
    device: str = 'cpu'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    fp16_quantization: bool = False

class ObjectDetectorConfig(BaseModel):
    log_level: LogLevel = LogLevel.WARNING
    model_config: YoloV8Config
    inference_size: tuple[int, int] = (640, 640)