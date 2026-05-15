import queue
import logging
import numpy as np
from models.model_config import SAMPLE_RATE, CHUNK_SECONDS, STRIDE_SECONDS

logger = logging.getLogger(__name__)

MIN_CHUNK_SAMPLES = int(0.1 * SAMPLE_RATE)  # 100ms minimum to avoid tiny clips


class AudioBufferManager:
    """Splits audio into overlapping chunks and queues them for ASR.

    Each instance is request-scoped (not shared globally) to prevent
    cross-request contamination.
    """

    def __init__(
        self,
        chunk_seconds: int = CHUNK_SECONDS,
        stride_seconds: int = STRIDE_SECONDS,
        sample_rate: int = SAMPLE_RATE,
    ):
        self._queue: queue.Queue = queue.Queue()
        self._chunk_samples = chunk_seconds * sample_rate
        self._stride_samples = stride_seconds * sample_rate
        self._sample_rate = sample_rate

    def enqueue(self, audio: np.ndarray) -> int:
        """Split audio into overlapping chunks and add to queue.

        Returns:
            Number of chunks enqueued.
        """
        if audio is None or len(audio) == 0:
            return 0

        n_chunks = 0
        start = 0
        while start < len(audio):
            end = min(start + self._chunk_samples, len(audio))
            chunk = audio[start:end]

            if len(chunk) >= MIN_CHUNK_SAMPLES:
                self._queue.put(chunk)
                n_chunks += 1

            if end >= len(audio):
                break
            start += self._stride_samples

        logger.debug("Enqueued %d chunks from %d samples", n_chunks, len(audio))
        return n_chunks

    def drain(self) -> list[np.ndarray]:
        """Return all queued chunks and clear the queue."""
        chunks = []
        while not self._queue.empty():
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    @property
    def size(self) -> int:
        return self._queue.qsize()
