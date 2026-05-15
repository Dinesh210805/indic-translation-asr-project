import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# transcribe_audio — local backend
# ---------------------------------------------------------------------------

class TestTranscribeAudio:
    def test_empty_array_returns_empty(self):
        from app.asr_pipeline import transcribe_audio
        result = transcribe_audio(np.array([], dtype=np.float32), "whisper-small (Local)")
        assert result == ""

    def test_none_returns_empty(self):
        from app.asr_pipeline import transcribe_audio
        result = transcribe_audio(None, "whisper-small (Local)")
        assert result == ""

    def test_calls_pipeline_and_returns_text(self):
        from app.asr_pipeline import transcribe_audio
        mock_pipeline = MagicMock(return_value={"text": "  வணக்கம்  "})
        with patch("app.asr_pipeline.load_asr_model", return_value=mock_pipeline):
            result = transcribe_audio(make_audio(2), "whisper-small (Local)")
        assert result == "வணக்கம்"

    def test_pipeline_returns_no_text_key(self):
        from app.asr_pipeline import transcribe_audio
        mock_pipeline = MagicMock(return_value={})
        with patch("app.asr_pipeline.load_asr_model", return_value=mock_pipeline):
            result = transcribe_audio(make_audio(2), "whisper-small (Local)")
        assert result == ""


# ---------------------------------------------------------------------------
# transcribe_chunks
# ---------------------------------------------------------------------------

class TestTranscribeChunks:
    def test_empty_list_returns_empty(self):
        from app.asr_pipeline import transcribe_chunks
        assert transcribe_chunks([], "whisper-small (Local)") == ""

    def test_joins_multiple_chunks(self):
        from app.asr_pipeline import transcribe_chunks
        responses = [{"text": "நன்றி"}, {"text": "வணக்கம்"}]
        mock_pipeline = MagicMock(side_effect=responses)
        with patch("app.asr_pipeline.load_asr_model", return_value=mock_pipeline):
            result = transcribe_chunks([make_audio(2), make_audio(2)], "whisper-small (Local)")
        assert "நன்றி" in result and "வணக்கம்" in result

    def test_skips_empty_chunk_results(self):
        from app.asr_pipeline import transcribe_chunks
        responses = [{"text": ""}, {"text": "தமிழ்"}]
        mock_pipeline = MagicMock(side_effect=responses)
        with patch("app.asr_pipeline.load_asr_model", return_value=mock_pipeline):
            result = transcribe_chunks([make_audio(2), make_audio(2)], "whisper-small (Local)")
        assert result == "தமிழ்"


# ---------------------------------------------------------------------------
# load_asr_model
# ---------------------------------------------------------------------------

class TestLoadAsr:
    def test_unknown_model_raises(self):
        from app.asr_pipeline import load_asr_model
        with pytest.raises(ValueError):
            load_asr_model("nonexistent-model")

    def test_groq_model_raises(self):
        from app.asr_pipeline import load_asr_model
        with pytest.raises(ValueError):
            load_asr_model("whisper-large-v3-turbo ⚡ Groq Cloud")

    def test_caches_model(self):
        from app.asr_pipeline import load_asr_model, _loaded_models
        mock_pipe = MagicMock()
        _loaded_models["whisper-small (Local)"] = mock_pipe
        result = load_asr_model("whisper-small (Local)")
        assert result is mock_pipe
        _loaded_models.pop("whisper-small (Local)", None)


# ---------------------------------------------------------------------------
# transcribe_with_groq
# ---------------------------------------------------------------------------

class TestTranscribeWithGroq:
    def test_raises_without_api_key(self):
        import importlib
        import app.asr_pipeline as mod
        original = mod._groq_client
        mod._groq_client = None
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(EnvironmentError):
                from app.asr_pipeline import transcribe_with_groq
                transcribe_with_groq("/tmp/audio.wav")
        mod._groq_client = original

    def test_calls_groq_api_and_returns_text(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF fake wav data")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  வணக்கம்  "

        with patch("app.asr_pipeline._get_groq_client", return_value=mock_client):
            from app.asr_pipeline import transcribe_with_groq
            result = transcribe_with_groq(str(audio_file))

        assert result == "வணக்கம்"
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_returns_empty_on_none_response(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF fake wav data")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = None

        with patch("app.asr_pipeline._get_groq_client", return_value=mock_client):
            from app.asr_pipeline import transcribe_with_groq
            result = transcribe_with_groq(str(audio_file))

        assert result == ""
