from pydantic import BaseModel
from enum import Enum

class ModelSizeEnum(str, Enum):
    NANO = 'n'
    SMALL = 's'
    MEDIUM = 'm'
    LARGE = 'l'
    XLARGE = 'x'

class YoloV8Config(BaseModel):
    size: ModelSizeEnum
    device: str

class ObjectDetectorConfig(BaseModel):
    model_config: YoloV8Config
    inference_size: tuple[int, int]