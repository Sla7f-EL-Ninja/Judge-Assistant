"""
tests/voice/conftest.py
------------------------
Shared fixtures and hooks for the Voice STT test suite.

Layout
------
tests/voice/
├── conftest.py          ← this file
├── test_samples/        ← real audio (optional, gitignored)
├── test_helpers.py      ← pure-function unit tests
├── test_stt_engine.py   ← STTEngine unit + integration tests
└── test_config.py       ← config-constant tests
"""

from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VOICE_TESTS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR     = VOICE_TESTS_DIR / "test_samples"
SAMPLES_DIR.mkdir(exist_ok=True)
PROJECT_ROOT    = VOICE_TESTS_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".flac", ".mp4", ".m4a"}


# ---------------------------------------------------------------------------
# CLI option & markers
# (try/except because tests/ocr/conftest.py may have already registered it)
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption(
            "--run-integration",
            action="store_true",
            default=False,
            help="Run integration tests that hit live Google STT API.",
        )
    except ValueError:
        pass  # already registered by another conftest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests requiring live Google STT API (--run-integration)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take > 10 s",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="Pass --run-integration to run live API tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_seconds: int, sample_rate: int = 16_000) -> bytes:
    """Build a silent WAV in memory.  No external deps — uses stdlib wave."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * sample_rate * duration_seconds * 2)
    return buf.getvalue()


@pytest.fixture(scope="session")
def synthetic_wav_bytes() -> bytes:
    """3-second silent WAV — triggers the single-call path (≤55 s)."""
    return _make_wav(3)


@pytest.fixture(scope="session")
def synthetic_long_wav_bytes() -> bytes:
    """65-second silent WAV — triggers the chunked parallel path (>55 s)."""
    return _make_wav(65)


@pytest.fixture(scope="session")
def synthetic_wav_file(tmp_path_factory, synthetic_wav_bytes) -> Path:
    """3 s WAV written to a temp file (for CLI / ingestion tests)."""
    p = tmp_path_factory.mktemp("audio") / "test_short.wav"
    p.write_bytes(synthetic_wav_bytes)
    return p


@pytest.fixture(scope="session")
def synthetic_long_wav_file(tmp_path_factory, synthetic_long_wav_bytes) -> Path:
    """65 s WAV written to a temp file (for integration long-path tests)."""
    p = tmp_path_factory.mktemp("audio") / "test_long.wav"
    p.write_bytes(synthetic_long_wav_bytes)
    return p


# ---------------------------------------------------------------------------
# Real-file fixtures  (fall back to synthetic when test_samples/ empty)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_audio_files() -> List[Path]:
    return sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


@pytest.fixture(scope="session")
def first_audio_bytes(sample_audio_files, synthetic_wav_bytes) -> bytes:
    """First real audio file as bytes, or the synthetic WAV if folder is empty."""
    if sample_audio_files:
        return sample_audio_files[0].read_bytes()
    return synthetic_wav_bytes


@pytest.fixture(scope="session")
def long_audio_bytes(sample_audio_files, synthetic_long_wav_bytes) -> bytes:
    """Largest real audio file, or the synthetic 65 s WAV if folder is empty."""
    if sample_audio_files:
        candidates = sorted(sample_audio_files, key=lambda p: p.stat().st_size, reverse=True)
        return candidates[0].read_bytes()
    return synthetic_long_wav_bytes


# ---------------------------------------------------------------------------
# Mock STT response builder
# ---------------------------------------------------------------------------

def make_stt_response(
    transcripts: list,
    confidences: list,
) -> MagicMock:
    """Build a mock SpeechClient.recognize() response.

    Parameters
    ----------
    transcripts:
        List of transcript strings, one per result.
    confidences:
        Parallel list of confidence values.
        Pass 0 or None to simulate a missing confidence score
        (the engine uses ``if alt.confidence:`` so falsy → omitted).
    """
    results = []
    for text, conf in zip(transcripts, confidences):
        alt = MagicMock()
        alt.transcript = text
        alt.confidence = conf or 0   # falsy → engine treats as absent
        result = MagicMock()
        result.alternatives = [alt]
        results.append(result)
    resp = MagicMock()
    resp.results = results
    return resp


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stt_engine_unit():
    """STTEngine with fully mocked internals — no network, no credentials."""
    from Voice.stt_engine import STTEngine

    engine = object.__new__(STTEngine)   # bypass __init__
    engine._speech = MagicMock()
    engine._client = MagicMock()
    engine._project_id = "test-project"
    engine._model = "chirp_2"
    engine._language_codes = ["ar-EG"]
    engine._max_workers = 4
    engine._recognizer = (
        "projects/test-project/locations/europe-west4/recognizers/_"
    )
    return engine


@pytest.fixture(scope="session")
def stt_engine():
    """Real STTEngine singleton — skips if credentials or project_id missing."""
    cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not cred or not Path(cred).exists():
        pytest.skip("GOOGLE_APPLICATION_CREDENTIALS not set or file missing")

    from config.voice import STT_PROJECT_ID
    if not STT_PROJECT_ID:
        pytest.skip(
            "STT project_id not configured. Set voice.google_stt.project_id "
            "in settings.yaml or JA_VOICE_GOOGLE_STT_PROJECT_ID in .env"
        )

    from Voice.stt_engine import get_stt_engine
    return get_stt_engine()
