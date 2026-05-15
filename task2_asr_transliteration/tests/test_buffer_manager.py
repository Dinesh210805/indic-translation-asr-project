import numpy as np
import pytest
from app.buffer_manager import AudioBufferManager


def make_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_enqueue_single_chunk():
    buf = AudioBufferManager()
    audio = make_audio(10)
    n = buf.enqueue(audio)
    assert n == 1
    assert buf.size == 1


def test_enqueue_splits_long_audio():
    buf = AudioBufferManager()
    audio = make_audio(75)  # 75s → 3 chunks at 30s with 5s stride
    n = buf.enqueue(audio)
    assert n == 3
    assert buf.size == 3


def test_drain_returns_all_chunks():
    buf = AudioBufferManager()
    audio = make_audio(75)
    buf.enqueue(audio)
    chunks = buf.drain()
    assert len(chunks) == 3
    assert buf.size == 0


def test_short_chunk_skipped():
    buf = AudioBufferManager()
    audio = make_audio(0.05)  # 50ms — below MIN_CHUNK_SAMPLES
    n = buf.enqueue(audio)
    assert n == 0


def test_empty_audio():
    buf = AudioBufferManager()
    audio = np.array([], dtype=np.float32)
    n = buf.enqueue(audio)
    assert n == 0
    chunks = buf.drain()
    assert chunks == []


def test_none_audio():
    buf = AudioBufferManager()
    n = buf.enqueue(None)
    assert n == 0


def test_drain_clears_queue():
    buf = AudioBufferManager()
    buf.enqueue(make_audio(10))
    buf.drain()
    assert buf.size == 0
    assert buf.drain() == []
