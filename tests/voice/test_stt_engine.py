"""
tests/voice/test_stt_engine.py
--------------------------------
Tests for Voice.stt_engine (STTEngine class, ffmpeg helpers, singleton).

Unit tests (no network, ffmpeg mocked or skipped):
  TestGuessSuffixEdge       — complementary edge cases beyond test_helpers.py
  TestGetDurationMs         — subprocess mock → correct parse / bad output
  TestExtractChunk          — subprocess mock → bytes returned / empty error
  TestTranscribeSingle      — mocked STT client → result structure
  TestTranscribeChunked     — mocked slice method → chunk count + ordering
  TestTranscribeDispatch    — empty bytes, short vs long path, error capture
  TestGetSttEngine          — missing project_id, singleton, thread safety

Integration tests (@pytest.mark.integration — live Google STT API):
  TestSTTEngineShortPath    — single synchronous call ≤55 s
  TestSTTEngineLongPath     — chunked parallel call >55 s
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, call, patch

import pytest

from conftest import make_stt_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"transcript", "confidence", "error"}


def _ok(transcript="النص المستخرج", confidence=0.92):
    return {"transcript": transcript, "confidence": confidence, "error": None}


def _err(msg="boom"):
    return {"transcript": "", "confidence": None, "error": msg}


# ---------------------------------------------------------------------------
# _get_duration_ms  (subprocess mock)
# ---------------------------------------------------------------------------

class TestGetDurationMs:
    _TARGET = "Voice.stt_engine.subprocess.run"

    def _run(self, ffmpeg_stderr: str) -> int:
        from Voice.stt_engine import _get_duration_ms
        mock_result = MagicMock()
        mock_result.stderr = ffmpeg_stderr.encode("utf-8")
        with patch("Voice.stt_engine._get_ffmpeg_exe", return_value="/ffmpeg"):
            with patch("Voice.stt_engine._write_temp", return_value="/tmp/a.wav"):
                with patch(self._TARGET, return_value=mock_result):
                    with patch("os.unlink"):
                        return _get_duration_ms(b"audio")

    def test_parses_seconds_correctly(self):
        ms = self._run("... Duration: 00:01:03.50, start: ...")
        assert ms == 63_500  # 63.5 s

    def test_parses_hours(self):
        ms = self._run("Duration: 01:00:00.00")
        assert ms == 3_600_000

    def test_parses_zero_duration(self):
        ms = self._run("Duration: 00:00:00.00")
        assert ms == 0

    def test_no_duration_in_output_raises(self):
        from Voice.stt_engine import _get_duration_ms
        mock_result = MagicMock()
        mock_result.stderr = b"some unrelated ffmpeg output"
        with patch("Voice.stt_engine._get_ffmpeg_exe", return_value="/ffmpeg"):
            with patch("Voice.stt_engine._write_temp", return_value="/tmp/x.wav"):
                with patch(self._TARGET, return_value=mock_result):
                    with patch("os.unlink"):
                        with pytest.raises(RuntimeError, match="duration"):
                            _get_duration_ms(b"audio")

    def test_temp_file_always_deleted(self):
        """Temp file must be cleaned up even when parsing succeeds."""
        mock_result = MagicMock()
        mock_result.stderr = b"Duration: 00:00:03.00"
        with patch("Voice.stt_engine._get_ffmpeg_exe", return_value="/ffmpeg"):
            with patch("Voice.stt_engine._write_temp", return_value="/tmp/x.wav"):
                with patch(self._TARGET, return_value=mock_result):
                    with patch("os.unlink") as mock_unlink:
                        from Voice.stt_engine import _get_duration_ms
                        _get_duration_ms(b"audio")
        mock_unlink.assert_called_once_with("/tmp/x.wav")


# ---------------------------------------------------------------------------
# _extract_chunk  (subprocess mock)
# ---------------------------------------------------------------------------

class TestExtractChunk:
    def _extract(self, stdout: bytes) -> bytes:
        from Voice.stt_engine import _extract_chunk
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.stderr = b""
        with patch("Voice.stt_engine._get_ffmpeg_exe", return_value="/ffmpeg"):
            with patch("Voice.stt_engine._write_temp", return_value="/tmp/x.wav"):
                with patch("Voice.stt_engine.subprocess.run", return_value=mock_result):
                    with patch("os.unlink"):
                        return _extract_chunk(b"audio", 0, 10_000)

    def test_returns_stdout_bytes(self):
        chunk = self._extract(b"\xff\xfbFAKEMP3")
        assert chunk == b"\xff\xfbFAKEMP3"

    def test_empty_stdout_raises(self):
        from Voice.stt_engine import _extract_chunk
        mock_result = MagicMock()
        mock_result.stdout = b""
        mock_result.stderr = b"some ffmpeg error"
        with patch("Voice.stt_engine._get_ffmpeg_exe", return_value="/ffmpeg"):
            with patch("Voice.stt_engine._write_temp", return_value="/tmp/x.wav"):
                with patch("Voice.stt_engine.subprocess.run", return_value=mock_result):
                    with patch("os.unlink"):
                        with pytest.raises(RuntimeError, match="ffmpeg"):
                            _extract_chunk(b"audio", 0, 10_000)

    def test_temp_file_always_deleted(self):
        mock_result = MagicMock()
        mock_result.stdout = b"data"
        with patch("Voice.stt_engine._get_ffmpeg_exe", return_value="/ffmpeg"):
            with patch("Voice.stt_engine._write_temp", return_value="/tmp/x.wav"):
                with patch("Voice.stt_engine.subprocess.run", return_value=mock_result):
                    with patch("os.unlink") as mock_unlink:
                        from Voice.stt_engine import _extract_chunk
                        _extract_chunk(b"audio", 0, 5_000)
        mock_unlink.assert_called_once_with("/tmp/x.wav")


# ---------------------------------------------------------------------------
# STTEngine._transcribe_single  (mocked client)
# ---------------------------------------------------------------------------

class TestTranscribeSingle:
    def test_returns_all_required_keys(self, stt_engine_unit):
        stt_engine_unit._client.recognize.return_value = make_stt_response(
            ["النص المستخرج"], [0.90]
        )
        result = stt_engine_unit._transcribe_single(b"audio", chunk_num=1)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_no_results_returns_empty(self, stt_engine_unit):
        resp = MagicMock()
        resp.results = []
        stt_engine_unit._client.recognize.return_value = resp
        result = stt_engine_unit._transcribe_single(b"audio")
        assert result == {"transcript": "", "confidence": None, "error": None}

    def test_single_result_with_confidence(self, stt_engine_unit):
        stt_engine_unit._client.recognize.return_value = make_stt_response(
            ["وفقا لنص المادة"], [0.93]
        )
        result = stt_engine_unit._transcribe_single(b"audio")
        assert result["transcript"] == "وفقا لنص المادة"
        assert abs(result["confidence"] - 0.93) < 1e-4
        assert result["error"] is None

    def test_multiple_results_joined(self, stt_engine_unit):
        stt_engine_unit._client.recognize.return_value = make_stt_response(
            ["جزء أول", "جزء ثان"], [0.90, 0.88]
        )
        result = stt_engine_unit._transcribe_single(b"audio")
        assert "جزء أول" in result["transcript"]
        assert "جزء ثان" in result["transcript"]

    def test_mean_confidence_computed(self, stt_engine_unit):
        stt_engine_unit._client.recognize.return_value = make_stt_response(
            ["a", "b"], [0.80, 0.90]
        )
        result = stt_engine_unit._transcribe_single(b"audio")
        assert abs(result["confidence"] - 0.85) < 0.001

    def test_zero_confidence_treated_as_absent(self, stt_engine_unit):
        """alt.confidence=0 is falsy — engine omits it → confidence=None."""
        stt_engine_unit._client.recognize.return_value = make_stt_response(
            ["text"], [0]
        )
        result = stt_engine_unit._transcribe_single(b"audio")
        assert result["confidence"] is None

    def test_exception_returns_error_result(self, stt_engine_unit):
        stt_engine_unit._client.recognize.side_effect = Exception("API down")
        result = stt_engine_unit._transcribe_single(b"audio")
        assert result["transcript"] == ""
        assert result["confidence"] is None
        assert "API down" in result["error"]

    def test_remove_repetitions_called_on_transcript(self, stt_engine_unit):
        """Hallucination filter must run on the joined transcript."""
        loop_text = " ".join(["ما"] * 50)
        stt_engine_unit._client.recognize.return_value = make_stt_response(
            [loop_text], [0.80]
        )
        result = stt_engine_unit._transcribe_single(b"audio")
        # 50 repeats → collapsed to 2
        assert result["transcript"].count("ما") == 2

    def test_content_request_uses_audio_bytes(self, stt_engine_unit):
        """The audio bytes must be passed as request.content."""
        resp = MagicMock(); resp.results = []
        stt_engine_unit._client.recognize.return_value = resp
        stt_engine_unit._transcribe_single(b"AUDIO_DATA")
        call_kwargs = stt_engine_unit._speech.RecognizeRequest.call_args[1]
        assert call_kwargs["content"] == b"AUDIO_DATA"


# ---------------------------------------------------------------------------
# STTEngine._transcribe_chunked  (mocked _transcribe_chunk_slice)
# ---------------------------------------------------------------------------

class TestTranscribeChunked:
    @pytest.mark.parametrize("duration_ms, overlap_ms, expected_chunks", [
        (61_000,  2_000, 2),   # 61 s  → [(0,52000),(50000,61000)]
        (110_000, 3_300, 3),   # 110 s → 3 chunks
        (55_001,  2_000, 2),   # just over limit → 2 chunks
        (100_000, 3_000, 2),   # 100 s → [(0,53000),(50000,100000)]
        (160_000, 4_800, 4),   # 160 s → 4 chunks
    ])
    def test_chunk_count(self, duration_ms, overlap_ms, expected_chunks,
                         stt_engine_unit):
        call_count = [0]
        def fake_slice(audio_bytes, start, end, num):
            call_count[0] += 1
            return _ok(f"chunk{num}", 0.90)

        with patch.object(stt_engine_unit, "_transcribe_chunk_slice",
                          side_effect=fake_slice):
            stt_engine_unit._transcribe_chunked(b"audio", duration_ms, overlap_ms)

        assert call_count[0] == expected_chunks

    def test_results_assembled_in_order(self, stt_engine_unit):
        """Output must be in chunk order, not future-completion order."""
        def fake_slice(audio_bytes, start, end, num):
            return _ok(transcript=f"chunk{num}", confidence=0.90)

        with patch.object(stt_engine_unit, "_transcribe_chunk_slice",
                          side_effect=fake_slice):
            result = stt_engine_unit._transcribe_chunked(b"audio", 110_000, 3_300)

        # "chunk1" must appear before "chunk2" before "chunk3"
        t = result["transcript"]
        assert t.index("chunk1") < t.index("chunk2") < t.index("chunk3")

    def test_all_chunks_fail_returns_error(self, stt_engine_unit):
        with patch.object(stt_engine_unit, "_transcribe_chunk_slice",
                          side_effect=lambda *a: _err("network")):
            result = stt_engine_unit._transcribe_chunked(b"audio", 61_000, 2_000)
        assert result["transcript"] == ""
        assert result["error"] is not None

    def test_partial_failure_returns_transcript(self, stt_engine_unit):
        """If ≥1 chunk succeeds with text, return that text, not an error."""
        call_num = [0]
        def fake_slice(audio_bytes, start, end, num):
            call_num[0] += 1
            if call_num[0] == 1:
                return _err("chunk 1 failed")
            return _ok(transcript="good text")

        with patch.object(stt_engine_unit, "_transcribe_chunk_slice",
                          side_effect=fake_slice):
            result = stt_engine_unit._transcribe_chunked(b"audio", 61_000, 2_000)

        assert "good text" in result["transcript"]
        assert result["error"] is None  # partial failure → no error at top level

    def test_mean_confidence_from_successful_chunks(self, stt_engine_unit):
        responses = [
            _ok(confidence=0.80),
            _ok(confidence=0.90),
        ]
        idx = [0]
        def fake_slice(*a):
            r = responses[idx[0]]
            idx[0] += 1
            return r

        with patch.object(stt_engine_unit, "_transcribe_chunk_slice",
                          side_effect=fake_slice):
            result = stt_engine_unit._transcribe_chunked(b"audio", 61_000, 2_000)

        assert abs(result["confidence"] - 0.85) < 0.001

    def test_all_none_confidence_gives_none(self, stt_engine_unit):
        with patch.object(stt_engine_unit, "_transcribe_chunk_slice",
                          side_effect=lambda *a: {"transcript": "x", "confidence": None, "error": None}):
            result = stt_engine_unit._transcribe_chunked(b"audio", 61_000, 2_000)
        assert result["confidence"] is None


# ---------------------------------------------------------------------------
# STTEngine.transcribe  (dispatch layer)
# ---------------------------------------------------------------------------

class TestTranscribeDispatch:
    def test_empty_bytes_returns_immediately(self, stt_engine_unit):
        result = stt_engine_unit.transcribe(b"")
        assert result == {"transcript": "", "confidence": None, "error": None}

    def test_empty_bytes_does_not_call_ffmpeg(self, stt_engine_unit):
        with patch("Voice.stt_engine._get_duration_ms") as mock_dur:
            stt_engine_unit.transcribe(b"")
        mock_dur.assert_not_called()

    def test_short_audio_calls_single_path(self, stt_engine_unit):
        with patch("Voice.stt_engine._get_duration_ms", return_value=30_000):
            with patch.object(stt_engine_unit, "_transcribe_single",
                              return_value=_ok()) as mock_single:
                stt_engine_unit.transcribe(b"audio")
        mock_single.assert_called_once()

    def test_long_audio_calls_chunked_path(self, stt_engine_unit):
        with patch("Voice.stt_engine._get_duration_ms", return_value=61_000):
            with patch.object(stt_engine_unit, "_transcribe_chunked",
                              return_value=_ok()) as mock_chunk:
                stt_engine_unit.transcribe(b"audio")
        mock_chunk.assert_called_once()

    def test_at_limit_uses_single_path(self, stt_engine_unit):
        """55 000 ms is exactly at the limit → single call."""
        with patch("Voice.stt_engine._get_duration_ms", return_value=55_000):
            with patch.object(stt_engine_unit, "_transcribe_single",
                              return_value=_ok()) as mock_single:
                stt_engine_unit.transcribe(b"audio")
        mock_single.assert_called_once()

    def test_one_ms_over_limit_uses_chunked_path(self, stt_engine_unit):
        """55 001 ms is just over → chunked path."""
        with patch("Voice.stt_engine._get_duration_ms", return_value=55_001):
            with patch.object(stt_engine_unit, "_transcribe_chunked",
                              return_value=_ok()) as mock_chunk:
                stt_engine_unit.transcribe(b"audio")
        mock_chunk.assert_called_once()

    def test_runtime_error_from_ffmpeg_captured(self, stt_engine_unit):
        with patch("Voice.stt_engine._get_duration_ms",
                   side_effect=RuntimeError("no ffmpeg output")):
            result = stt_engine_unit.transcribe(b"audio")
        assert result["transcript"] == ""
        assert result["error"] is not None
        assert "ffmpeg" in result["error"].lower() or "no" in result["error"]

    def test_generic_exception_captured(self, stt_engine_unit):
        with patch("Voice.stt_engine._get_duration_ms",
                   side_effect=ValueError("unexpected")):
            result = stt_engine_unit.transcribe(b"audio")
        assert result["error"] is not None

    def test_result_always_has_required_keys(self, stt_engine_unit):
        with patch("Voice.stt_engine._get_duration_ms", return_value=3_000):
            with patch.object(stt_engine_unit, "_transcribe_single",
                              return_value=_ok()):
                result = stt_engine_unit.transcribe(b"audio")
        assert REQUIRED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# get_stt_engine  (singleton factory)
# ---------------------------------------------------------------------------

class TestGetSttEngine:
    def _reset(self):
        import Voice.stt_engine as mod
        mod._engine_instance = None

    def test_missing_project_id_raises(self):
        self._reset()
        with patch("config.voice.STT_PROJECT_ID", ""):
            with pytest.raises(RuntimeError, match="project_id"):
                from Voice.stt_engine import get_stt_engine
                get_stt_engine()
        self._reset()

    def test_returns_stt_engine_instance(self):
        self._reset()
        with patch("config.voice.STT_PROJECT_ID", "my-project"):
            with patch("Voice.stt_engine.STTEngine") as MockEngine:
                MockEngine.return_value = MagicMock()
                from Voice.stt_engine import get_stt_engine
                engine = get_stt_engine()
        assert engine is not None
        self._reset()

    def test_singleton_same_object(self):
        self._reset()
        with patch("config.voice.STT_PROJECT_ID", "my-project"):
            with patch("Voice.stt_engine.STTEngine", return_value=MagicMock()):
                from Voice.stt_engine import get_stt_engine
                e1 = get_stt_engine()
                e2 = get_stt_engine()
        assert e1 is e2
        self._reset()

    def test_thread_safe_singleton(self):
        """8 concurrent threads must all receive the same instance."""
        self._reset()
        instances = []
        lock = threading.Lock()

        def grab():
            with patch("config.voice.STT_PROJECT_ID", "proj"):
                with patch("Voice.stt_engine.STTEngine", return_value=MagicMock()):
                    from Voice.stt_engine import get_stt_engine
                    e = get_stt_engine()
                    with lock:
                        instances.append(id(e))

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(instances)) == 1, (
            f"Singleton built {len(set(instances))} distinct objects — must be 1"
        )
        self._reset()


# ---------------------------------------------------------------------------
# Integration tests  (live Google STT API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestSTTEngineShortPath:
    """Single synchronous call ≤55 s with real Google Chirp 2."""

    def test_result_has_required_keys(self, stt_engine, synthetic_wav_bytes):
        result = stt_engine.transcribe(synthetic_wav_bytes)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_no_error_on_valid_audio(self, stt_engine, synthetic_wav_bytes):
        result = stt_engine.transcribe(synthetic_wav_bytes)
        assert result["error"] is None

    def test_transcript_is_string(self, stt_engine, synthetic_wav_bytes):
        result = stt_engine.transcribe(synthetic_wav_bytes)
        assert isinstance(result["transcript"], str)

    def test_confidence_in_range_or_none(self, stt_engine, synthetic_wav_bytes):
        result = stt_engine.transcribe(synthetic_wav_bytes)
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_bytes_short_circuits(self, stt_engine):
        result = stt_engine.transcribe(b"")
        assert result == {"transcript": "", "confidence": None, "error": None}

    def test_real_audio_if_available(self, stt_engine, first_audio_bytes):
        result = stt_engine.transcribe(first_audio_bytes)
        assert result["error"] is None
        assert isinstance(result["transcript"], str)


@pytest.mark.integration
@pytest.mark.slow
class TestSTTEngineLongPath:
    """Chunked parallel path >55 s with real Google Chirp 2 + ffmpeg."""

    def test_result_has_required_keys(self, stt_engine, synthetic_long_wav_bytes):
        result = stt_engine.transcribe(synthetic_long_wav_bytes)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_no_error_on_long_audio(self, stt_engine, synthetic_long_wav_bytes):
        result = stt_engine.transcribe(synthetic_long_wav_bytes)
        assert result["error"] is None

    def test_transcript_is_string(self, stt_engine, synthetic_long_wav_bytes):
        result = stt_engine.transcribe(synthetic_long_wav_bytes)
        assert isinstance(result["transcript"], str)

    def test_real_long_audio_if_available(self, stt_engine, long_audio_bytes):
        result = stt_engine.transcribe(long_audio_bytes)
        assert result["error"] is None
        assert isinstance(result["transcript"], str)
