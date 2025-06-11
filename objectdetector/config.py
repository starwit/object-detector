from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource


class YoloConfig(BaseModel):
    weights_path: Path
    auto_download: bool = False
    device: str = 'cpu'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    fp16_quantization: bool = False
    nms_agnostic: bool = False


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_ids: Annotated[List[str], Field(min_length=1)]
    input_stream_prefix: str = 'videosource'
    output_stream_prefix: str = 'objectdetector'


class ObjectDetectorConfig(BaseSettings):
    log_level: LogLevel = LogLevel.WARNING
    model: YoloConfig
    inference_size: tuple[int, int] = (640, 640)
    classes: Optional[List[int]] = None
    max_batch_size: Annotated[int, Field(ge=1)] = 1
    max_batch_interval: Annotated[float, Field(ge=0)] = 0
    drop_edge_detections: bool = False
    redis: RedisConfig
    prometheus_port: Annotated[int, Field(gt=1024, le=65536)] = 8000

    model_config = SettingsConfigDict(env_nested_delimiter='__')

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)