import time
from typing import Iterable, List, NamedTuple

from prometheus_client import Histogram

BATCH_SIZE = Histogram('object_detector_batch_size', 'The size of the batches being processed',
                       buckets=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32))

class BatchEntry(NamedTuple):
    stream_key: str
    proto_data: bytes

def batched(iterable: Iterable, max_batch_size: int, max_batch_interval: float) -> Iterable[List[BatchEntry]]:
    batch = []
    prev_batch_time = 0
    for stream_key, proto_data in iterable:
        if stream_key is None:
            yield []
            continue
        batch.append(BatchEntry(stream_key, proto_data))
        if len(batch) >= max_batch_size or time.time() - prev_batch_time > max_batch_interval:
            BATCH_SIZE.observe(len(batch))
            prev_batch_time = time.time()
            yield batch
            batch = []