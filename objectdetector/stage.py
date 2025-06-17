import logging
import signal
import threading

from prometheus_client import Counter, Histogram, start_http_server
from visionlib.pipeline.consumer import RedisConsumer
from visionlib.pipeline.publisher import RedisPublisher

from .batch import batched
from .config import ObjectDetectorConfig
from .detector import Detector

logger = logging.getLogger(__name__)

REDIS_PUBLISH_DURATION = Histogram('object_detector_redis_publish_duration', 'The time it takes to push a message onto the Redis stream',
                                   buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
FRAME_COUNTER = Counter('object_detector_frame_counter', 'How many frames have been consumed from the Redis input stream')

def run_stage():

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

    logger.setLevel(CONFIG.log_level.value)

    logger.info(f'Starting prometheus metrics endpoint on port {CONFIG.prometheus_port}')

    start_http_server(CONFIG.prometheus_port)

    logger.info(f'Starting object detector stage. Config: {CONFIG.model_dump_json(indent=2)}')

    detector = Detector(CONFIG)

    consume = RedisConsumer(CONFIG.redis.host, CONFIG.redis.port, 
                            stream_keys=[f'{CONFIG.redis.input_stream_prefix}:{id}' for id in CONFIG.redis.stream_ids],
                            block=int(CONFIG.max_batch_interval * 1000))
    publish = RedisPublisher(CONFIG.redis.host, CONFIG.redis.port)
    
    with consume, publish:
        for batch in batched(consume(), CONFIG.max_batch_size, CONFIG.max_batch_interval):
            if stop_event.is_set():
                break

            if len(batch) == 0:
                continue

            FRAME_COUNTER.inc(len(batch))

            output_batch = detector.get(batch)

            if output_batch is None:
                continue
            
            for entry in output_batch:
                stream_id = entry.stream_key.split(':')[-1]
                with REDIS_PUBLISH_DURATION.time():
                    publish(f'{CONFIG.redis.output_stream_prefix}:{stream_id}', entry.proto_data)
            