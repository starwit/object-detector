import signal
import redis
import threading
from objectdetector.config import ObjectDetectorConfig
from objectdetector.detector import Detector
from typing import List

def extract_stream_id(input_stream_name: str) -> str:
    return input_stream_name.split(':')[1]


class RedisListener:

    def __init__(self, redis_client: redis.Redis, stream_ids: List[str]) -> None:
        self._redis_client = redis_client
        self._stream_ids = stream_ids

        self._last_retrieved_id: str = None
        self._current_stream_idx: int = 0

    def get_new_message(self):
        result = redis_conn.xread(
            count=1,
            block=2000,
            streams={f'videosource:{id}': '$' if self._last_retrieved_id is None else self._last_retrieved_id 
                        for id in CONFIG.redis.stream_ids}
        )
        
        if result is None or len(result) == 0:
            return None
        
        self._last_retrieved_id = result[0][1][0][0].decode('utf-8')

        return result

    def _incr_stream_idx(self):
        self._current_stream_idx = (self._current_stream_idx + 1) % len(self._stream_ids)


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
    
    redis_conn = redis.Redis(
        host=CONFIG.redis.host,
        port=CONFIG.redis.port,
    )
    
    redis_listener = RedisListener(redis_conn, CONFIG.redis.stream_ids)

    # Start processing images
    while not stop_event.is_set():

        reply = redis_listener.get_new_message()

        if reply is None:
            continue

        for item in reply:
            if stop_event.is_set():
                break
            
            # These unpacking incantations are ugly but necessary due to redis reply structure
            input_proto = item[1][0][1][b'proto_data']
            input_stream = item[0].decode('utf-8')

            output_proto = detector.get(input_proto)

            if output_proto is None:
                continue

            redis_conn.xadd(name=f'objectdetector:{extract_stream_id(input_stream)}', fields={'proto_data': output_proto}, maxlen=10)
