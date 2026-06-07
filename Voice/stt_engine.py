# """
# Voice.stt_engine
# -----------------
# Google Speech-to-Text V2 engine using Chirp 2.

# Long audio handling
# -------------------
# Audio longer than 55 s is split into 50-second chunks.  Each chunk
# extends past the boundary by an adaptive overlap — 3% of total audio
# duration, capped between 2 s and 5 s.  Only trailing overlap is applied
# (not both trailing and leading), so the actual overlap region is exactly
# overlap_ms, not 2× it.

# Example — 61 s recording:
#     overlap = max(2s, min(61*0.03, 5s)) = 2s
#     Chunk 1: 0s → 52s  (50s content + 2s tail)
#     Chunk 2: 50s → 61s
#     Overlap region: 50s–52s = 2s  ✓

# All chunks are transcribed in parallel via ThreadPoolExecutor.
# Seam deduplication (difflib.SequenceMatcher) removes boundary duplicates.

# Audio processing
# ----------------
# Uses the ffmpeg binary bundled with imageio-ffmpeg via subprocess.
# Audio is written to a NamedTemporaryFile for probing and chunking
# because several formats (OGG, MP4, WebM) require seeking.
# Temp files are always deleted in finally blocks.

# Dependencies (pip install only — no system packages):
#     imageio-ffmpeg

# pydub is no longer needed:
#     pip uninstall pydub
# """

# from __future__ import annotations

# import difflib
# import logging
# import os
# import re
# import subprocess
# import tempfile
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Optional

# from dotenv import load_dotenv

# load_dotenv()

# logger = logging.getLogger(__name__)

# _engine_instance: Optional["STTEngine"] = None
# _engine_lock = threading.Lock()

# _SYNC_LIMIT_MS = 55_000     # audio <= this → single synchronous call
# _CHUNK_MS = 50_000          # chunk step size (unique content per chunk)


# # ---------------------------------------------------------------------------
# # Adaptive overlap
# # ---------------------------------------------------------------------------

# def _compute_overlap_ms(duration_ms: int) -> int:
#     """Return overlap in ms: 3% of total duration, clamped 2s–5s.

#     Trailing-only overlap — actual overlap region = exactly this value.

#     For the expected 0–5 min range:
#         61 s  →  1.83 s  →  2 s  (floor)
#        120 s  →  3.6 s
#        180 s  →  5.4 s  →  5 s  (cap)
#        300 s  →  9.0 s  →  5 s  (cap)
#     """
#     overlap_s = max(2.0, min(duration_ms / 1000 * 0.03, 5.0))
#     return int(overlap_s * 1000)


# # ---------------------------------------------------------------------------
# # FFmpeg helpers
# # ---------------------------------------------------------------------------

# def _get_ffmpeg_exe() -> str:
#     try:
#         import imageio_ffmpeg  # noqa: PLC0415
#         return imageio_ffmpeg.get_ffmpeg_exe()
#     except ImportError:
#         raise RuntimeError(
#             "imageio-ffmpeg is not installed. Run: pip install imageio-ffmpeg"
#         )


# def _guess_suffix(audio_bytes: bytes) -> str:
#     """Detect format from magic bytes for reliable temp file naming."""
#     if audio_bytes[:4] == b"OggS":
#         return ".ogg"
#     if audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3"):
#         return ".mp3"
#     if audio_bytes[:4] == b"RIFF":
#         return ".wav"
#     if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
#         return ".webm"
#     if audio_bytes[:4] == b"fLaC":
#         return ".flac"
#     if len(audio_bytes) > 8 and audio_bytes[4:8] == b"ftyp":
#         return ".mp4"
#     return ".audio"


# def _write_temp(audio_bytes: bytes) -> str:
#     """Write audio bytes to a NamedTemporaryFile and return its path."""
#     suffix = _guess_suffix(audio_bytes)
#     fd, path = tempfile.mkstemp(suffix=suffix)
#     try:
#         with os.fdopen(fd, "wb") as f:
#             f.write(audio_bytes)
#     except Exception:
#         try:
#             os.unlink(path)
#         except OSError:
#             pass
#         raise
#     return path


# def _get_duration_ms(audio_bytes: bytes) -> int:
#     """Get audio duration in ms by probing with bundled ffmpeg."""
#     ffmpeg = _get_ffmpeg_exe()
#     tmp_path = _write_temp(audio_bytes)
#     try:
#         result = subprocess.run(
#             [ffmpeg, "-i", tmp_path, "-f", "null", os.devnull],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
#         stderr = result.stderr.decode("utf-8", errors="replace")
#         match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr)
#         if not match:
#             raise RuntimeError(
#                 f"Could not determine audio duration.\n"
#                 f"ffmpeg output:\n{stderr[:800]}"
#             )
#         hours, minutes, seconds = match.groups()
#         total_s = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
#         return int(total_s * 1000)
#     finally:
#         try:
#             os.unlink(tmp_path)
#         except OSError:
#             pass


# def _extract_chunk(audio_bytes: bytes, start_ms: int, end_ms: int) -> bytes:
#     """Extract a time slice from audio, returned as MP3 bytes."""
#     ffmpeg = _get_ffmpeg_exe()
#     start_s = start_ms / 1000
#     duration_s = (end_ms - start_ms) / 1000
#     tmp_path = _write_temp(audio_bytes)
#     try:
#         result = subprocess.run(
#             [
#                 ffmpeg, "-y",
#                 "-i", tmp_path,
#                 "-ss", f"{start_s:.3f}",
#                 "-t", f"{duration_s:.3f}",
#                 "-f", "mp3",
#                 "-acodec", "libmp3lame",
#                 "-q:a", "4",
#                 "pipe:1",
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
#         if not result.stdout:
#             stderr = result.stderr.decode("utf-8", errors="replace")
#             raise RuntimeError(
#                 f"ffmpeg chunk extraction failed:\n{stderr[:400]}"
#             )
#         return result.stdout
#     finally:
#         try:
#             os.unlink(tmp_path)
#         except OSError:
#             pass


# # ---------------------------------------------------------------------------
# # Seam deduplication
# # ---------------------------------------------------------------------------

# def _merge_transcripts(transcripts: list[str]) -> str:
#     """Join overlapping chunk transcripts without duplicating boundary words.

#     Inspects a 30-word window on each seam.  If SequenceMatcher finds a
#     match of >= 2 words, the duplicate is trimmed.  If no match (silence
#     at boundary), chunks are appended directly.
#     """
#     if not transcripts:
#         return ""
#     if len(transcripts) == 1:
#         return transcripts[0].strip()

#     WINDOW = 30
#     result_words = transcripts[0].split()

#     for i in range(1, len(transcripts)):
#         next_words = transcripts[i].split()
#         if not next_words:
#             continue

#         tail = result_words[-WINDOW:]
#         head = next_words[:WINDOW]

#         matcher = difflib.SequenceMatcher(None, tail, head, autojunk=False)
#         match = matcher.find_longest_match(0, len(tail), 0, len(head))

#         if match.size >= 2:
#             keep = len(result_words) - WINDOW + match.a
#             skip = match.b + match.size
#             result_words = result_words[:keep] + next_words[skip:]
#             logger.debug(
#                 "Seam %d/%d: matched %d words — trimmed cleanly",
#                 i, len(transcripts) - 1, match.size,
#             )
#         else:
#             result_words.extend(next_words)
#             logger.debug(
#                 "Seam %d/%d: no match — appended directly",
#                 i, len(transcripts) - 1,
#             )

#     return " ".join(result_words).strip()


# # ---------------------------------------------------------------------------
# # Engine
# # ---------------------------------------------------------------------------

# class STTEngine:
#     """Wrapper around Google Speech-to-Text V2 (Chirp 2).

#     Short audio  (≤ 55 s) → single synchronous recognize() call.
#     Long audio   (> 55 s) → adaptive-overlap chunks, parallel STT, seam merge.
#     """

#     def __init__(
#         self,
#         project_id: str,
#         model: str,
#         language_codes: list[str],
#         max_workers: int = 4,
#         location: str = "europe-west4",
#     ) -> None:
#         from google.cloud.speech_v2 import SpeechClient  # noqa: PLC0415
#         from google.cloud.speech_v2.types import cloud_speech  # noqa: PLC0415
#         from google.api_core.client_options import ClientOptions  # noqa: PLC0415

#         self._speech = cloud_speech
#         self._client = SpeechClient(
#             client_options=ClientOptions(
#                 api_endpoint=f"{location}-speech.googleapis.com"
#             )
#         )
#         self._project_id = project_id
#         self._model = model
#         self._language_codes = language_codes
#         self._max_workers = max_workers
#         self._recognizer = (
#             f"projects/{project_id}/locations/{location}/recognizers/_"
#         )

#         logger.info(
#             "STTEngine initialised (model=%s, languages=%s, location=%s, workers=%d)",
#             model, language_codes, location, max_workers,
#         )

#     # ------------------------------------------------------------------
#     # Public
#     # ------------------------------------------------------------------

#     def transcribe(self, audio_bytes: bytes) -> dict:
#         """Transcribe raw audio bytes to Arabic text.

#         Returns
#         -------
#         dict: transcript (str), confidence (float|None), error (str|None)
#         """
#         if not audio_bytes:
#             return {"transcript": "", "confidence": None, "error": None}

#         try:
#             duration_ms = _get_duration_ms(audio_bytes)
#             logger.info("Audio duration: %.1fs", duration_ms / 1000)

#             if duration_ms <= _SYNC_LIMIT_MS:
#                 logger.info("Single-call path (%.1fs)", duration_ms / 1000)
#                 return self._transcribe_single(audio_bytes, chunk_num=1)

#             overlap_ms = _compute_overlap_ms(duration_ms)
#             logger.info(
#                 "Chunked path — duration=%.1fs, overlap=%.1fs",
#                 duration_ms / 1000, overlap_ms / 1000,
#             )
#             return self._transcribe_chunked(audio_bytes, duration_ms, overlap_ms)

#         except RuntimeError as exc:
#             logger.error("Audio processing error: %s", exc)
#             return {"transcript": "", "confidence": None, "error": str(exc)}

#         except Exception as exc:
#             logger.exception("STT engine error: %s", exc)
#             return {"transcript": "", "confidence": None, "error": str(exc)}

#     # ------------------------------------------------------------------
#     # Chunked parallel path
#     # ------------------------------------------------------------------

#     def _transcribe_chunked(
#         self, audio_bytes: bytes, duration_ms: int, overlap_ms: int
#     ) -> dict:
#         """Split audio into chunks with trailing overlap and transcribe in parallel.

#         Chunk boundaries step by _CHUNK_MS.  Each chunk extends past its
#         boundary by overlap_ms (trailing only).  The next chunk starts at
#         the clean boundary — so actual overlap = exactly overlap_ms.
#         """
#         boundaries: list[tuple[int, int]] = []
#         start_ms = 0
#         while start_ms < duration_ms:
#             chunk_start = start_ms
#             chunk_end = min(start_ms + _CHUNK_MS + overlap_ms, duration_ms)
#             boundaries.append((chunk_start, chunk_end))
#             start_ms += _CHUNK_MS

#         n = len(boundaries)
#         logger.info(
#             "%d chunk(s): %s",
#             n,
#             ", ".join(
#                 f"{s/1000:.1f}s–{e/1000:.1f}s"
#                 for s, e in boundaries
#             ),
#         )

#         raw_transcripts: list[str] = [""] * n
#         confidences: list[float] = []
#         errors: list[str] = []

#         with ThreadPoolExecutor(max_workers=min(n, self._max_workers)) as pool:
#             future_to_idx = {
#                 pool.submit(
#                     self._transcribe_chunk_slice,
#                     audio_bytes, start, end, i + 1,
#                 ): i
#                 for i, (start, end) in enumerate(boundaries)
#             }
#             for future in as_completed(future_to_idx):
#                 idx = future_to_idx[future]
#                 try:
#                     result = future.result()
#                     raw_transcripts[idx] = result.get("transcript", "")
#                     if result.get("confidence") is not None:
#                         confidences.append(result["confidence"])
#                     if result.get("error"):
#                         errors.append(f"chunk {idx + 1}: {result['error']}")
#                 except Exception as exc:
#                     logger.error("Chunk %d future error: %s", idx + 1, exc)
#                     errors.append(f"chunk {idx + 1}: {exc}")

#         full_transcript = _merge_transcripts(
#             [t for t in raw_transcripts if t]
#         )
#         mean_conf = (
#             round(sum(confidences) / len(confidences), 4)
#             if confidences else None
#         )

#         if errors and not full_transcript:
#             return {"transcript": "", "confidence": None, "error": "; ".join(errors)}

#         if errors:
#             logger.warning(
#                 "Partial chunk failures (%d/%d): %s",
#                 len(errors), n, "; ".join(errors),
#             )

#         logger.info(
#             "Chunked transcription complete: %d chars from %d/%d chunks",
#             len(full_transcript), n - len(errors), n,
#         )

#         return {"transcript": full_transcript, "confidence": mean_conf, "error": None}

#     def _transcribe_chunk_slice(
#         self,
#         audio_bytes: bytes,
#         start_ms: int,
#         end_ms: int,
#         chunk_num: int,
#     ) -> dict:
#         chunk = _extract_chunk(audio_bytes, start_ms, end_ms)
#         return self._transcribe_single(chunk, chunk_num=chunk_num)

#     # ------------------------------------------------------------------
#     # Single API call
#     # ------------------------------------------------------------------

#     def _transcribe_single(self, audio_bytes: bytes, chunk_num: int = 1) -> dict:
#         try:
#             config = self._speech.RecognitionConfig(
#                 auto_decoding_config=self._speech.AutoDetectDecodingConfig(),
#                 language_codes=self._language_codes,
#                 model=self._model,
#             )
#             request = self._speech.RecognizeRequest(
#                 recognizer=self._recognizer,
#                 config=config,
#                 content=audio_bytes,
#             )
#             response = self._client.recognize(request=request)

#             if not response.results:
#                 logger.debug("Chunk %d: no results (silence)", chunk_num)
#                 return {"transcript": "", "confidence": None, "error": None}

#             transcripts: list[str] = []
#             confidences: list[float] = []

#             for result in response.results:
#                 if result.alternatives:
#                     alt = result.alternatives[0]
#                     transcripts.append(alt.transcript)
#                     if alt.confidence:
#                         confidences.append(alt.confidence)

#             transcript = " ".join(transcripts).strip()
#             confidence = (
#                 round(sum(confidences) / len(confidences), 4)
#                 if confidences else None
#             )

#             logger.debug(
#                 "Chunk %d: %d chars, conf=%s",
#                 chunk_num, len(transcript), confidence,
#             )
#             return {"transcript": transcript, "confidence": confidence, "error": None}

#         except Exception as exc:
#             logger.error("Chunk %d STT error: %s", chunk_num, exc)
#             return {"transcript": "", "confidence": None, "error": str(exc)}


# # ---------------------------------------------------------------------------
# # Convenience wrapper
# # ---------------------------------------------------------------------------

# def transcribe_audio(audio_bytes: bytes) -> dict:
#     return get_stt_engine().transcribe(audio_bytes)


# # ---------------------------------------------------------------------------
# # Singleton factory
# # ---------------------------------------------------------------------------

# def get_stt_engine() -> STTEngine:
#     global _engine_instance
#     if _engine_instance is None:
#         with _engine_lock:
#             if _engine_instance is None:
#                 from config.voice import (  # noqa: PLC0415
#                     STT_LANGUAGE_CODES,
#                     STT_LOCATION,
#                     STT_MAX_WORKERS,
#                     STT_MODEL,
#                     STT_PROJECT_ID,
#                 )

#                 if not STT_PROJECT_ID:
#                     raise RuntimeError(
#                         "STT_PROJECT_ID is not set. Add 'project_id' under "
#                         "'voice.google_stt' in settings.yaml or set "
#                         "JA_VOICE_GOOGLE_STT_PROJECT_ID in your .env file."
#                     )

#                 _engine_instance = STTEngine(
#                     project_id=STT_PROJECT_ID,
#                     model=STT_MODEL,
#                     language_codes=STT_LANGUAGE_CODES,
#                     max_workers=STT_MAX_WORKERS,
#                     location=STT_LOCATION,
#                 )
#     return _engine_instance


"""
Voice.stt_engine
-----------------
Google Speech-to-Text V2 engine using Chirp 2.

Long audio handling
-------------------
Audio longer than 55 s is split into 50-second chunks.  Each chunk
extends past the boundary by an adaptive overlap — 3% of total audio
duration, capped between 2 s and 5 s.  Only trailing overlap is applied
(not both trailing and leading), so the actual overlap region is exactly
overlap_ms, not 2× it.

Example — 61 s recording:
    overlap = max(2s, min(61*0.03, 5s)) = 2s
    Chunk 1: 0s → 52s  (50s content + 2s tail)
    Chunk 2: 50s → 61s
    Overlap region: 50s–52s = 2s  ✓

All chunks are transcribed in parallel via ThreadPoolExecutor.
Seam deduplication (difflib.SequenceMatcher) removes boundary duplicates.

Hallucination filtering
-----------------------
STT models sometimes enter repetition loops on silence or low-quality
audio sections (e.g. "ما فيش" × 150).  _remove_repetitions() collapses
consecutive repeated n-grams (n=1 and n=2) down to max_repeats copies.

Audio processing
----------------
Uses the ffmpeg binary bundled with imageio-ffmpeg via subprocess.
Audio is written to a NamedTemporaryFile for probing and chunking
because several formats (OGG, MP4, WebM) require seeking.
Temp files are always deleted in finally blocks.

Dependencies (pip install only — no system packages):
    imageio-ffmpeg

pydub is no longer needed:
    pip uninstall pydub
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_engine_instance: Optional["STTEngine"] = None
_engine_lock = threading.Lock()

_SYNC_LIMIT_MS = 55_000     # audio <= this → single synchronous call
_CHUNK_MS = 50_000          # chunk step size (unique content per chunk)


# ---------------------------------------------------------------------------
# Adaptive overlap
# ---------------------------------------------------------------------------

def _compute_overlap_ms(duration_ms: int) -> int:
    """Return overlap in ms: 3% of total duration, clamped 2s–5s.

    Trailing-only overlap — actual overlap region = exactly this value.

    For the expected 0–5 min range:
        61 s  →  1.83 s  →  2 s  (floor)
       120 s  →  3.6 s
       180 s  →  5.4 s  →  5 s  (cap)
       300 s  →  9.0 s  →  5 s  (cap)
    """
    overlap_s = max(2.0, min(duration_ms / 1000 * 0.03, 5.0))
    return int(round(overlap_s * 1000))


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "imageio-ffmpeg is not installed. Run: pip install imageio-ffmpeg"
        )


def _guess_suffix(audio_bytes: bytes) -> str:
    """Detect format from magic bytes for reliable temp file naming."""
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return ".mp3"
    if audio_bytes[:4] == b"RIFF":
        return ".wav"
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if len(audio_bytes) > 8 and audio_bytes[4:8] == b"ftyp":
        return ".mp4"
    return ".audio"


def _write_temp(audio_bytes: bytes) -> str:
    """Write audio bytes to a NamedTemporaryFile and return its path."""
    suffix = _guess_suffix(audio_bytes)
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _get_duration_ms(audio_bytes: bytes) -> int:
    """Get audio duration in ms by probing with bundled ffmpeg."""
    ffmpeg = _get_ffmpeg_exe()
    tmp_path = _write_temp(audio_bytes)
    try:
        result = subprocess.run(
            [ffmpeg, "-i", tmp_path, "-f", "null", os.devnull],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr)
        if not match:
            raise RuntimeError(
                f"Could not determine audio duration.\n"
                f"ffmpeg output:\n{stderr[:800]}"
            )
        hours, minutes, seconds = match.groups()
        total_s = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return int(total_s * 1000)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_chunk(audio_bytes: bytes, start_ms: int, end_ms: int) -> bytes:
    """Extract a time slice from audio, returned as MP3 bytes."""
    ffmpeg = _get_ffmpeg_exe()
    start_s = start_ms / 1000
    duration_s = (end_ms - start_ms) / 1000
    tmp_path = _write_temp(audio_bytes)
    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", tmp_path,
                "-ss", f"{start_s:.3f}",
                "-t", f"{duration_s:.3f}",
                "-f", "mp3",
                "-acodec", "libmp3lame",
                "-q:a", "4",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not result.stdout:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg chunk extraction failed:\n{stderr[:400]}"
            )
        return result.stdout
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Hallucination filtering
# ---------------------------------------------------------------------------

def _dedupe_ngram(words: list[str], n: int, max_repeats: int) -> list[str]:
    """Cap consecutive repetitions of any n-word sequence to max_repeats."""
    result: list[str] = []
    i = 0
    while i < len(words):
        if i + n > len(words):
            result.extend(words[i:])
            break
        ngram = words[i: i + n]
        count = 1
        while i + (count + 1) * n <= len(words):
            if words[i + count * n: i + (count + 1) * n] == ngram:
                count += 1
            else:
                break
        kept = min(count, max_repeats)
        for _ in range(kept):
            result.extend(ngram)
        i += count * n
    return result


def _remove_repetitions(text: str, max_repeats: int = 2) -> str:
    """Remove hallucinated repetition loops from STT output.

    Collapses consecutive repeated n-grams for n=1 (single words) and
    n=2 (two-word phrases) down to max_repeats copies each.

    Examples:
        "ما ما ما ما"              → "ما ما"
        "ما فيش ما فيش ما فيش"    → "ما فيش ما فيش"

    max_repeats=2 preserves legitimate emphasis (e.g. "ما فيش ما فيش"
    is natural in Egyptian Arabic) while eliminating 150-repeat loops.
    """
    if not text:
        return text
    words = text.split()
    if len(words) < 2:
        return text

    original_len = len(words)
    for ngram_size in (1, 2):
        words = _dedupe_ngram(words, ngram_size, max_repeats)

    cleaned = " ".join(words)
    if len(words) < original_len:
        logger.warning(
            "Repetition hallucination removed: %d words → %d words",
            original_len, len(words),
        )
    return cleaned


# ---------------------------------------------------------------------------
# Seam deduplication
# ---------------------------------------------------------------------------

def _merge_transcripts(transcripts: list[str]) -> str:
    """Join overlapping chunk transcripts without duplicating boundary words.

    Inspects a 30-word window on each seam.  If SequenceMatcher finds a
    match of >= 2 words, the duplicate is trimmed.  If no match (silence
    at boundary), chunks are appended directly.
    """
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0].strip()

    WINDOW = 30
    result_words = transcripts[0].split()

    for i in range(1, len(transcripts)):
        next_words = transcripts[i].split()
        if not next_words:
            continue

        tail = result_words[-WINDOW:]
        head = next_words[:WINDOW]

        matcher = difflib.SequenceMatcher(None, tail, head, autojunk=False)
        match = matcher.find_longest_match(0, len(tail), 0, len(head))

        if match.size >= 2:
            # Safely find the absolute start index of the tail window
            tail_start = max(0, len(result_words) - WINDOW)
            
            # Keep everything in chunk 1 *before* the overlap starts
            keep = tail_start + match.a
            
            # Keep everything in chunk 2 *from* the overlap onwards
            skip = match.b
            
            result_words = result_words[:keep] + next_words[skip:]
            logger.debug(
                "Seam %d/%d: matched %d words — trimmed cleanly",
                i, len(transcripts) - 1, match.size,
            )
        else:
            result_words.extend(next_words)
            logger.debug(
                "Seam %d/%d: no match — appended directly",
                i, len(transcripts) - 1,
            )

    return " ".join(result_words).strip()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class STTEngine:
    """Wrapper around Google Speech-to-Text V2 (Chirp 2).

    Short audio  (≤ 55 s) → single synchronous recognize() call.
    Long audio   (> 55 s) → adaptive-overlap chunks, parallel STT, seam merge.
    """

    def __init__(
        self,
        project_id: str,
        model: str,
        language_codes: list[str],
        max_workers: int = 4,
        location: str = "europe-west4",
    ) -> None:
        from google.cloud.speech_v2 import SpeechClient  # noqa: PLC0415
        from google.cloud.speech_v2.types import cloud_speech  # noqa: PLC0415
        from google.api_core.client_options import ClientOptions  # noqa: PLC0415

        self._speech = cloud_speech
        self._client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com"
            )
        )
        self._project_id = project_id
        self._model = model
        self._language_codes = language_codes
        self._max_workers = max_workers
        self._recognizer = (
            f"projects/{project_id}/locations/{location}/recognizers/_"
        )

        logger.info(
            "STTEngine initialised (model=%s, languages=%s, location=%s, workers=%d)",
            model, language_codes, location, max_workers,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes) -> dict:
        """Transcribe raw audio bytes to Arabic text.

        Returns
        -------
        dict: transcript (str), confidence (float|None), error (str|None)
        """
        if not audio_bytes:
            return {"transcript": "", "confidence": None, "error": None}

        try:
            duration_ms = _get_duration_ms(audio_bytes)
            logger.info("Audio duration: %.1fs", duration_ms / 1000)

            if duration_ms <= _SYNC_LIMIT_MS:
                logger.info("Single-call path (%.1fs)", duration_ms / 1000)
                return self._transcribe_single(audio_bytes, chunk_num=1)

            overlap_ms = _compute_overlap_ms(duration_ms)
            logger.info(
                "Chunked path — duration=%.1fs, overlap=%.1fs",
                duration_ms / 1000, overlap_ms / 1000,
            )
            return self._transcribe_chunked(audio_bytes, duration_ms, overlap_ms)

        except RuntimeError as exc:
            logger.error("Audio processing error: %s", exc)
            return {"transcript": "", "confidence": None, "error": str(exc)}

        except Exception as exc:
            logger.exception("STT engine error: %s", exc)
            return {"transcript": "", "confidence": None, "error": str(exc)}

    # ------------------------------------------------------------------
    # Chunked parallel path
    # ------------------------------------------------------------------

    def _transcribe_chunked(
        self, audio_bytes: bytes, duration_ms: int, overlap_ms: int
    ) -> dict:
        """Split audio into chunks with trailing overlap and transcribe in parallel."""
        boundaries: list[tuple[int, int]] = []
        start_ms = 0
        while start_ms < duration_ms:
            chunk_start = start_ms
            chunk_end = min(start_ms + _CHUNK_MS + overlap_ms, duration_ms)
            boundaries.append((chunk_start, chunk_end))
            start_ms += _CHUNK_MS

        n = len(boundaries)
        logger.info(
            "%d chunk(s): %s",
            n,
            ", ".join(
                f"{s/1000:.1f}s–{e/1000:.1f}s"
                for s, e in boundaries
            ),
        )

        raw_transcripts: list[str] = [""] * n
        confidences: list[float] = []
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=min(n, self._max_workers)) as pool:
            future_to_idx = {
                pool.submit(
                    self._transcribe_chunk_slice,
                    audio_bytes, start, end, i + 1,
                ): i
                for i, (start, end) in enumerate(boundaries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    raw_transcripts[idx] = result.get("transcript", "")
                    if result.get("confidence") is not None:
                        confidences.append(result["confidence"])
                    if result.get("error"):
                        errors.append(f"chunk {idx + 1}: {result['error']}")
                except Exception as exc:
                    logger.error("Chunk %d future error: %s", idx + 1, exc)
                    errors.append(f"chunk {idx + 1}: {exc}")

        full_transcript = _merge_transcripts(
            [t for t in raw_transcripts if t]
        )
        mean_conf = (
            round(sum(confidences) / len(confidences), 4)
            if confidences else None
        )

        if errors and not full_transcript:
            return {"transcript": "", "confidence": None, "error": "; ".join(errors)}

        if errors:
            logger.warning(
                "Partial chunk failures (%d/%d): %s",
                len(errors), n, "; ".join(errors),
            )

        logger.info(
            "Chunked transcription complete: %d chars from %d/%d chunks",
            len(full_transcript), n - len(errors), n,
        )

        return {"transcript": full_transcript, "confidence": mean_conf, "error": None}

    def _transcribe_chunk_slice(
        self,
        audio_bytes: bytes,
        start_ms: int,
        end_ms: int,
        chunk_num: int,
    ) -> dict:
        chunk = _extract_chunk(audio_bytes, start_ms, end_ms)
        return self._transcribe_single(chunk, chunk_num=chunk_num)

    # ------------------------------------------------------------------
    # Single API call
    # ------------------------------------------------------------------

    def _transcribe_single(self, audio_bytes: bytes, chunk_num: int = 1) -> dict:
        try:
            config = self._speech.RecognitionConfig(
                auto_decoding_config=self._speech.AutoDetectDecodingConfig(),
                language_codes=self._language_codes,
                model=self._model,
            )
            request = self._speech.RecognizeRequest(
                recognizer=self._recognizer,
                config=config,
                content=audio_bytes,
            )
            response = self._client.recognize(request=request)

            if not response.results:
                logger.debug("Chunk %d: no results (silence)", chunk_num)
                return {"transcript": "", "confidence": None, "error": None}

            transcripts: list[str] = []
            confidences: list[float] = []

            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    transcripts.append(alt.transcript)
                    if alt.confidence:
                        confidences.append(alt.confidence)

            transcript = " ".join(transcripts).strip()
            transcript = _remove_repetitions(transcript)

            confidence = (
                round(sum(confidences) / len(confidences), 4)
                if confidences else None
            )

            logger.debug(
                "Chunk %d: %d chars, conf=%s",
                chunk_num, len(transcript), confidence,
            )
            return {"transcript": transcript, "confidence": confidence, "error": None}

        except Exception as exc:
            logger.error("Chunk %d STT error: %s", chunk_num, exc)
            return {"transcript": "", "confidence": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes) -> dict:
    return get_stt_engine().transcribe(audio_bytes)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_stt_engine() -> STTEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                from config.voice import (  # noqa: PLC0415
                    STT_LANGUAGE_CODES,
                    STT_LOCATION,
                    STT_MAX_WORKERS,
                    STT_MODEL,
                    STT_PROJECT_ID,
                )

                if not STT_PROJECT_ID:
                    raise RuntimeError(
                        "STT_PROJECT_ID is not set. Add 'project_id' under "
                        "'voice.google_stt' in settings.yaml or set "
                        "JA_VOICE_GOOGLE_STT_PROJECT_ID in your .env file."
                    )

                _engine_instance = STTEngine(
                    project_id=STT_PROJECT_ID,
                    model=STT_MODEL,
                    language_codes=STT_LANGUAGE_CODES,
                    max_workers=STT_MAX_WORKERS,
                    location=STT_LOCATION,
                )
    return _engine_instance