import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, BaseSettings, conint, conlist


def yaml_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
    return yaml.load(Path(os.environ.get('SETTINGS_FILE', 'settings.yaml')).read_text('utf-8'), Loader=yaml.Loader)


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


class RedisConfig(BaseModel):
    host: str
    port: conint(ge=1, le=65536)
    video_source_ids: conlist(str)


class ObjectDetectorConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    model_config: YoloV8Config
    inference_size: tuple[int, int] = (640, 640)
    classes: conlist(int) = None
    redis: RedisConfig = None

    class Config:
        env_nested_delimiter = '__'
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (init_settings, env_settings, yaml_config_settings_source, file_secret_settings)