import signal
import threading
from typing import List

import redis
from visionlib.pipeline.consumer import RedisConsumer
from visionlib.pipeline.publisher import RedisPublisher

from objectdetector.config import ObjectDetectorConfig
from objectdetector.detector import Detector


def extract_stream_id(input_stream_name: str) -> str:
    return input_stream_name.split(':')[1]

if __name__ == '__main__':

    stop_event = threading.Event()

    # Register signal handlers
    def sig_handler(signum, _):
        signame = signal.Signals(signum).name
        print(f'Caught signal {signame} ({signum}). Exiting...')
        stop_event.set()

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    # Load config from settings.yaml / env vars
    CONFIG = ObjectDetectorConfig()

    detector = Detector(CONFIG)

    publish = RedisPublisher(CONFIG.redis.host, CONFIG.redis.port)
    consume = RedisConsumer(CONFIG.redis.host, CONFIG.redis.port, stream_keys=CONFIG.redis.stream_ids)
    
    with publish, consume:
        for stream_key, proto_data in consume():
            if stream_key is None:
                continue

            output_proto_data = detector.get(proto_data)

            if output_proto_data is None:
                continue

            publish(stream_key, proto_data)

            if stop_event.is_set():
                break
            