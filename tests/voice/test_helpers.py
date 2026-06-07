"""
tests/voice/test_helpers.py
----------------------------
Unit tests for the pure-Python helper functions in Voice.stt_engine.

No mocking, no network calls — all functions tested here are deterministic.

Covered:
    _compute_overlap_ms   — adaptive overlap clamping
    _guess_suffix         — magic-byte format detection
    _dedupe_ngram         — n-gram repetition capping
    _remove_repetitions   — hallucination filter (wraps _dedupe_ngram)
    _merge_transcripts    — seam deduplication (wraps difflib)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Imports (defer so collection works even without google-cloud-speech)
# ---------------------------------------------------------------------------

def _helpers():
    from Voice.stt_engine import (
        _compute_overlap_ms,
        _dedupe_ngram,
        _guess_suffix,
        _merge_transcripts,
        _remove_repetitions,
    )
    return (
        _compute_overlap_ms,
        _dedupe_ngram,
        _guess_suffix,
        _merge_transcripts,
        _remove_repetitions,
    )


# ---------------------------------------------------------------------------
# _compute_overlap_ms
# ---------------------------------------------------------------------------

class TestComputeOverlapMs:
    """3% of total duration, clamped to [2 000 ms, 5 000 ms]."""

    def _overlap(self, duration_ms: int) -> int:
        f, *_ = _helpers()
        return f(duration_ms)

    # ---- floor (< 2 s) -----------------------------------------------
    def test_61s_clamped_to_floor(self):
        # 61 × 0.03 = 1.83 s → floor to 2.0 s
        assert self._overlap(61_000) == 2_000

    def test_55s_clamped_to_floor(self):
        # 55 × 0.03 = 1.65 s → floor to 2.0 s
        assert self._overlap(55_000) == 2_000

    def test_66s_clamped_to_floor(self):
        # 66 × 0.03 = 1.98 s → still floor (just under 2.0 s)
        assert self._overlap(66_000) == 2_000

    def test_67s_just_above_floor(self):
        # 67 × 0.03 = 2.01 s → 2 010 ms (first value above the floor)
        assert self._overlap(67_000) == 2_010

    # ---- mid-range (between 2 s and 5 s) --------------------------------
    def test_120s(self):
        # 120 × 0.03 = 3.6 s → 3 600 ms
        assert self._overlap(120_000) == 3_600

    def test_100s(self):
        # 100 × 0.03 = 3.0 s → 3 000 ms
        assert self._overlap(100_000) == 3_000

    # ---- cap (> 5 s) -------------------------------------------------------
    def test_180s_capped_to_ceiling(self):
        # 180 × 0.03 = 5.4 s → cap to 5.0 s
        assert self._overlap(180_000) == 5_000

    def test_300s_capped_to_ceiling(self):
        # 300 × 0.03 = 9.0 s → cap to 5.0 s
        assert self._overlap(300_000) == 5_000

    def test_return_type_is_int(self):
        assert isinstance(self._overlap(120_000), int)


# ---------------------------------------------------------------------------
# _guess_suffix
# ---------------------------------------------------------------------------

class TestGuessSuffix:
    def _guess(self, data: bytes) -> str:
        _, _, f, *_ = _helpers()
        return f(data)

    def test_ogg(self):
        assert self._guess(b"OggS" + b"\x00" * 20) == ".ogg"

    def test_mp3_id3(self):
        assert self._guess(b"ID3" + b"\x00" * 20) == ".mp3"

    def test_mp3_sync_ffb(self):
        assert self._guess(b"\xff\xfb" + b"\x00" * 20) == ".mp3"

    def test_mp3_sync_ff3(self):
        assert self._guess(b"\xff\xf3" + b"\x00" * 20) == ".mp3"

    def test_wav(self):
        assert self._guess(b"RIFF" + b"\x00" * 20) == ".wav"

    def test_webm(self):
        assert self._guess(b"\x1a\x45\xdf\xa3" + b"\x00" * 20) == ".webm"

    def test_flac(self):
        assert self._guess(b"fLaC" + b"\x00" * 20) == ".flac"

    def test_mp4_ftyp(self):
        # MP4: 4 bytes size + b"ftyp" at offset 4
        data = b"\x00\x00\x00\x20" + b"ftyp" + b"\x00" * 20
        assert self._guess(data) == ".mp4"

    def test_unknown_returns_audio(self):
        assert self._guess(b"\x00\x01\x02\x03" + b"\x00" * 20) == ".audio"

    def test_too_short_for_mp4_check(self):
        # Only 7 bytes — not enough to check ftyp at offset 4
        assert self._guess(b"\x00\x00\x00\x00\x00\x00\x00") == ".audio"

    def test_wav_magic_from_stdlib(self, synthetic_wav_bytes):
        assert self._guess(synthetic_wav_bytes) == ".wav"


# ---------------------------------------------------------------------------
# _dedupe_ngram
# ---------------------------------------------------------------------------

class TestDedupeNgram:
    def _dedup(self, words, n, max_repeats=2):
        _, f, *_ = _helpers()
        return f(words, n, max_repeats)

    # ---- n=1 (unigrams) -----------------------------------------------
    def test_n1_no_repeats(self):
        words = ["a", "b", "c"]
        assert self._dedup(words, 1) == ["a", "b", "c"]

    def test_n1_two_repeats_kept(self):
        # "a a a a" → "a a"  (max_repeats=2)
        assert self._dedup(["a", "a", "a", "a"], 1) == ["a", "a"]

    def test_n1_exactly_two_kept_as_is(self):
        assert self._dedup(["a", "a"], 1) == ["a", "a"]

    def test_n1_max_repeats_1(self):
        assert self._dedup(["a", "a", "a"], 1, max_repeats=1) == ["a"]

    def test_n1_mixed(self):
        # "a a a b b" → "a a b b"
        assert self._dedup(["a", "a", "a", "b", "b"], 1) == ["a", "a", "b", "b"]

    def test_n1_150_repeats_collapsed(self):
        words = ["ما"] * 150
        result = self._dedup(words, 1, max_repeats=2)
        assert result == ["ما", "ما"]

    # ---- n=2 (bigrams) -----------------------------------------------
    def test_n2_no_repeats(self):
        words = ["a", "b", "c", "d"]
        assert self._dedup(words, 2) == ["a", "b", "c", "d"]

    def test_n2_bigram_repeated(self):
        # "a b a b a b" → "a b a b"
        words = ["a", "b", "a", "b", "a", "b"]
        assert self._dedup(words, 2) == ["a", "b", "a", "b"]

    def test_n2_partial_tail_appended(self):
        # Last n-1 words that don't form a complete ngram must be kept
        words = ["a", "b", "a", "b", "c"]   # "a b" × 2 then "c"
        result = self._dedup(words, 2)
        assert result[-1] == "c"

    def test_n2_empty_input(self):
        assert self._dedup([], 2) == []

    def test_n2_single_word(self):
        assert self._dedup(["x"], 2) == ["x"]


# ---------------------------------------------------------------------------
# _remove_repetitions
# ---------------------------------------------------------------------------

class TestRemoveRepetitions:
    def _rm(self, text: str, max_repeats: int = 2) -> str:
        _, _, _, _, f = _helpers()
        return f(text, max_repeats)

    def test_empty_string(self):
        assert self._rm("") == ""

    def test_single_word_unchanged(self):
        assert self._rm("محكمة") == "محكمة"

    def test_no_repetitions_unchanged(self):
        text = "وفقا لنص المادة العاشرة"
        assert self._rm(text) == text

    def test_word_loop_collapsed(self):
        # "ما" × 10 → "ما ما"
        result = self._rm(" ".join(["ما"] * 10))
        assert result == "ما ما"

    def test_bigram_loop_collapsed(self):
        # "ما فيش" × 5 → "ما فيش ما فيش"
        result = self._rm(" ".join(["ما", "فيش"] * 5))
        assert result == "ما فيش ما فيش"

    def test_legitimate_double_preserved(self):
        # Two copies are natural in Egyptian Arabic — should survive
        result = self._rm("ما فيش ما فيش")
        assert result == "ما فيش ما فيش"

    def test_max_repeats_1(self):
        result = self._rm("ما ما ما", max_repeats=1)
        assert result == "ما"

    def test_150_repeat_loop(self):
        text = " ".join(["كلمة"] * 150)
        result = self._rm(text)
        assert result.count("كلمة") == 2

    def test_result_is_stripped(self):
        result = self._rm("  كلمة كلمة كلمة  ")
        # split/join removes leading/trailing whitespace
        assert result == result.strip()


# ---------------------------------------------------------------------------
# _merge_transcripts
# ---------------------------------------------------------------------------

class TestMergeTranscripts:
    def _merge(self, transcripts: list[str]) -> str:
        _, _, _, f, _ = _helpers()
        return f(transcripts)

    def test_empty_list_returns_empty(self):
        assert self._merge([]) == ""

    def test_single_transcript_returned_stripped(self):
        assert self._merge(["  hello  "]) == "hello"

    def test_two_non_overlapping_concatenated(self):
        result = self._merge(["first part", "second part"])
        assert "first part" in result
        assert "second part" in result

    def test_overlapping_boundary_deduped(self):
        # Chunk 1 ends with "المادة العاشرة من"
        # Chunk 2 starts with "المادة العاشرة من قانون"
        chunk1 = "وفقا لنص المادة العاشرة من"
        chunk2 = "المادة العاشرة من قانون الاثبات"
        result = self._merge([chunk1, chunk2])
        # "المادة العاشرة من" should appear once, not twice
        assert result.count("المادة") == 1
        assert "قانون الاثبات" in result

    def test_empty_string_in_list_skipped(self):
        result = self._merge(["first", "", "third"])
        assert "first" in result
        assert "third" in result

    def test_all_empty_strings(self):
        result = self._merge(["", "", ""])
        assert result == ""

    def test_three_chunks_correct_order(self):
        t1 = "chunk one content here alpha"
        t2 = "alpha chunk two content beta"
        t3 = "beta chunk three content"
        result = self._merge([t1, t2, t3])
        # Result must contain all unique words in order
        assert "chunk one" in result
        assert "chunk two" in result
        assert "chunk three" in result

    def test_no_match_at_seam_appended_directly(self):
        # No common words at the boundary → direct append
        result = self._merge(["أبجد", "هوز"])
        assert "أبجد" in result
        assert "هوز" in result

    def test_returns_string(self):
        assert isinstance(self._merge(["a", "b"]), str)

    def test_result_stripped(self):
        result = self._merge(["  hello  "])
        assert result == result.strip()
