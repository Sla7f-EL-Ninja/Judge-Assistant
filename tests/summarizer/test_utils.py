"""
test_utils.py — Unit tests for Summerize/utils.py

Tests:
    T-UTILS-01: Hamza normalization (أ/إ/آ → ا)
    T-UTILS-02: Alef Maksura normalization (ى → ي)
    T-UTILS-03: escape_braces() double-brace encoding
    T-UTILS-04: llm_invoke_with_retry — succeeds on second attempt
    T-UTILS-05: llm_invoke_with_retry — immediate raise on non-transient error
    T-UTILS-06: llm_invoke_with_retry — raises after all retries exhausted
"""

import pathlib
import sys
from unittest.mock import MagicMock, call

import pytest


from summarize.utils import escape_braces, llm_invoke_with_retry, normalize_arabic_for_matching


@pytest.mark.unit
class TestNormalizeArabicForMatching:
    def test_hamza_alef_above(self):
        """T-UTILS-01: أ → ا"""
        assert normalize_arabic_for_matching("أحمد") == "احمد"

    def test_hamza_alef_below(self):
        """T-UTILS-01: إ → ا"""
        assert normalize_arabic_for_matching("إسلام") == "اسلام"

    def test_hamza_madda(self):
        """T-UTILS-01: آ → ا"""
        assert normalize_arabic_for_matching("آخر") == "اخر"

    def test_all_three_hamza_variants(self):
        """T-UTILS-01: all three Hamza variants in one string."""
        result = normalize_arabic_for_matching("أحمد إبراهيم آخر")
        assert result == "احمد ابراهيم اخر"

    def test_alef_maksura(self):
        """T-UTILS-02: ى → ي"""
        assert normalize_arabic_for_matching("على") == "علي"

    def test_alef_maksura_in_name(self):
        """T-UTILS-02: Alef Maksura in name 'مصطفى'."""
        assert normalize_arabic_for_matching("مصطفى") == "مصطفي"

    def test_combined_hamza_and_alef_maksura(self):
        """T-UTILS-02: Both Hamza and Alef Maksura normalized together."""
        result = normalize_arabic_for_matching("على مصطفى")
        assert result == "علي مصطفي"

    def test_non_arabic_unchanged(self):
        """Non-Arabic characters pass through unchanged."""
        assert normalize_arabic_for_matching("abc 123") == "abc 123"

    def test_empty_string(self):
        assert normalize_arabic_for_matching("") == ""

    def test_idempotent(self):
        """Applying twice gives same result as applying once."""
        text = "أحمد إبراهيم آخر على مصطفى"
        once = normalize_arabic_for_matching(text)
        twice = normalize_arabic_for_matching(once)
        assert once == twice


@pytest.mark.unit
class TestEscapeBraces:
    def test_single_braces_escaped(self):
        """T-UTILS-03: { and } are doubled."""
        assert escape_braces("text {var} end") == "text {{var}} end"

    def test_no_braces_unchanged(self):
        assert escape_braces("plain text") == "plain text"

    def test_only_open_brace(self):
        assert escape_braces("{") == "{{"

    def test_only_close_brace(self):
        assert escape_braces("}") == "}}"

    def test_arabic_with_braces(self):
        result = escape_braces("النص {متغير} هنا")
        assert result == "النص {{متغير}} هنا"

    def test_empty_string(self):
        assert escape_braces("") == ""

    def test_format_safe_after_escape(self):
        """After escape_braces, str.format() does not raise KeyError."""
        text = "value is {0} and {key}"
        escaped = escape_braces(text)
        formatted = escaped.format()  # No positional or keyword args
        assert "{0}" in formatted
        assert "{key}" in formatted


@pytest.mark.unit
class TestLlmInvokeWithRetry:
    def test_succeeds_on_first_attempt(self):
        """Happy path: single invocation returns result."""
        parser = MagicMock()
        parser.invoke.return_value = "result"
        result = llm_invoke_with_retry(parser, "messages", max_retries=2, base_delay=0.0)
        assert result == "result"
        assert parser.invoke.call_count == 1

    def test_retries_on_transient_error_succeeds_on_second(self):
        """T-UTILS-04: Transient error on first call, success on second."""
        parser = MagicMock()
        parser.invoke.side_effect = [
            RuntimeError("rate limit exceeded"),
            "success",
        ]
        result = llm_invoke_with_retry(parser, "messages", max_retries=2, base_delay=0.0)
        assert result == "success"
        assert parser.invoke.call_count == 2

    def test_immediate_raise_on_non_transient_error(self):
        """T-UTILS-05: Non-transient error raises immediately without retry."""
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("authentication failed")
        with pytest.raises(RuntimeError, match="authentication failed"):
            llm_invoke_with_retry(parser, "messages", max_retries=2, base_delay=0.0)
        assert parser.invoke.call_count == 1

    def test_raises_after_all_retries_exhausted(self):
        """T-UTILS-06: Transient error on all 3 attempts (max_retries=2) → raises."""
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("rate limit exceeded")
        with pytest.raises(RuntimeError, match="rate limit"):
            llm_invoke_with_retry(parser, "messages", max_retries=2, base_delay=0.0)
        assert parser.invoke.call_count == 3  # max_retries + 1

    def test_timeout_signal_is_transient(self):
        """'timeout' in error message is treated as transient."""
        parser = MagicMock()
        parser.invoke.side_effect = [
            RuntimeError("connection timeout"),
            "ok",
        ]
        result = llm_invoke_with_retry(parser, "messages", max_retries=1, base_delay=0.0)
        assert result == "ok"

    def test_429_status_is_transient(self):
        """'429' in error message treated as transient (rate limit HTTP code)."""
        parser = MagicMock()
        parser.invoke.side_effect = [
            RuntimeError("HTTP 429"),
            "ok",
        ]
        result = llm_invoke_with_retry(parser, "messages", max_retries=1, base_delay=0.0)
        assert result == "ok"

    def test_schema_error_not_transient(self):
        """Schema validation errors are not retried."""
        parser = MagicMock()
        parser.invoke.side_effect = ValueError("schema validation error")
        with pytest.raises(ValueError):
            llm_invoke_with_retry(parser, "messages", max_retries=2, base_delay=0.0)
        assert parser.invoke.call_count == 1

    def test_max_retries_zero(self):
        """max_retries=0 means single attempt only."""
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("rate limit")
        with pytest.raises(RuntimeError):
            llm_invoke_with_retry(parser, "messages", max_retries=0, base_delay=0.0)
        assert parser.invoke.call_count == 1
