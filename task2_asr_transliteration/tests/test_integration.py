"""Integration tests for the full process_audio pipeline.

All external calls (model loading, Groq API, librosa) are mocked so these
tests run without any GPU, audio files, or API keys.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(tmp_path, name: str = "audio.wav") -> str:
    p = tmp_path / name
    p.write_bytes(b"RIFF fake wav data")
    return str(p)


# ---------------------------------------------------------------------------
# Local backend path
# ---------------------------------------------------------------------------

class TestLocalBackendPath:
    def test_happy_path_local(self, tmp_path):
        """Full local path: load audio → buffer → transcribe → transliterate."""
        wav = _make_wav(tmp_path)
        fake_audio = np.zeros(16000 * 5, dtype=np.float32)

        with (
            patch("app.interface.load_audio", return_value=fake_audio),
            patch("app.interface.transcribe_chunks", return_value="வணக்கம்"),
            patch("app.interface.save_output_json", return_value="/tmp/out.json"),
        ):
            from app.interface import process_audio
            transcript, romanized, json_path = process_audio(
                wav, "whisper-small (Local)", "ITRANS"
            )

        assert transcript == "வணக்கம்"
        assert isinstance(romanized, str) and len(romanized) > 0
        assert json_path == "/tmp/out.json"

    def test_empty_transcript_returns_message(self, tmp_path):
        wav = _make_wav(tmp_path)
        fake_audio = np.zeros(16000 * 5, dtype=np.float32)

        with (
            patch("app.interface.load_audio", return_value=fake_audio),
            patch("app.interface.transcribe_chunks", return_value=""),
        ):
            from app.interface import process_audio
            transcript, romanized, json_path = process_audio(
                wav, "whisper-small (Local)", "ITRANS"
            )

        assert "Could not transcribe" in transcript
        assert romanized == ""
        assert json_path is None

    def test_audio_load_error_returns_message(self, tmp_path):
        wav = _make_wav(tmp_path)

        with patch("app.interface.load_audio", side_effect=Exception("librosa error")):
            from app.interface import process_audio
            transcript, romanized, json_path = process_audio(
                wav, "whisper-small (Local)", "ITRANS"
            )

        assert "Error loading audio" in transcript
        assert json_path is None

    def test_none_audio_path(self):
        from app.interface import process_audio
        transcript, romanized, json_path = process_audio(None, "whisper-small (Local)", "ITRANS")
        assert transcript == "No audio provided."
        assert json_path is None


# ---------------------------------------------------------------------------
# Groq backend path
# ---------------------------------------------------------------------------

class TestGroqBackendPath:
    def test_happy_path_groq(self, tmp_path):
        """Full Groq path: skip buffer, call Groq API, transliterate."""
        wav = _make_wav(tmp_path)

        with (
            patch("app.interface.transcribe_with_groq", return_value="தமிழ் நாடு"),
            patch("app.interface.save_output_json", return_value="/tmp/groq_out.json"),
        ):
            from app.interface import process_audio
            transcript, romanized, json_path = process_audio(
                wav, "whisper-large-v3-turbo ⚡ Groq Cloud", "ITRANS"
            )

        assert transcript == "தமிழ் நாடு"
        assert isinstance(romanized, str) and len(romanized) > 0
        assert json_path == "/tmp/groq_out.json"

    def test_groq_missing_api_key_returns_config_error(self, tmp_path):
        wav = _make_wav(tmp_path)

        with patch(
            "app.interface.transcribe_with_groq",
            side_effect=EnvironmentError("GROQ_API_KEY is not set"),
        ):
            from app.interface import process_audio
            transcript, romanized, json_path = process_audio(
                wav, "whisper-large-v3-turbo ⚡ Groq Cloud", "ITRANS"
            )

        assert "Configuration error" in transcript
        assert json_path is None

    def test_groq_api_error_returns_transcription_error(self, tmp_path):
        wav = _make_wav(tmp_path)

        with patch(
            "app.interface.transcribe_with_groq",
            side_effect=Exception("401 Unauthorized"),
        ):
            from app.interface import process_audio
            transcript, romanized, json_path = process_audio(
                wav, "whisper-large-v3-turbo ⚡ Groq Cloud", "ITRANS"
            )

        assert "Transcription error" in transcript
        assert json_path is None

    def test_groq_path_does_not_call_load_audio(self, tmp_path):
        """Confirm load_audio is never invoked for Groq backend."""
        wav = _make_wav(tmp_path)

        with (
            patch("app.interface.transcribe_with_groq", return_value="test"),
            patch("app.interface.save_output_json", return_value=None),
            patch("app.interface.load_audio") as mock_load,
        ):
            from app.interface import process_audio
            process_audio(wav, "whisper-large-v3-turbo ⚡ Groq Cloud", "ITRANS")

        mock_load.assert_not_called()
