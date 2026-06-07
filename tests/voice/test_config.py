"""
tests/voice/test_config.py
---------------------------
Unit tests for config.voice constants and _resolve_project_id().

All tests are pure-Python — no network, no real credentials needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Type + value checks for exported constants
# ---------------------------------------------------------------------------

class TestVoiceConfigConstants:

    def test_stt_model_is_string(self):
        from config.voice import STT_MODEL
        assert isinstance(STT_MODEL, str)

    def test_stt_model_value(self):
        from config.voice import STT_MODEL
        assert STT_MODEL == "chirp_2"

    def test_language_codes_is_list(self):
        from config.voice import STT_LANGUAGE_CODES
        assert isinstance(STT_LANGUAGE_CODES, list)

    def test_language_codes_contains_arabic(self):
        from config.voice import STT_LANGUAGE_CODES
        assert "ar-EG" in STT_LANGUAGE_CODES

    def test_location_is_string(self):
        from config.voice import STT_LOCATION
        assert isinstance(STT_LOCATION, str)

    def test_location_value(self):
        from config.voice import STT_LOCATION
        assert STT_LOCATION == "europe-west4"

    def test_max_workers_is_positive_int(self):
        from config.voice import STT_MAX_WORKERS
        assert isinstance(STT_MAX_WORKERS, int)
        assert STT_MAX_WORKERS >= 1

    def test_chunk_duration_is_positive_int(self):
        from config.voice import STT_CHUNK_DURATION_SECONDS
        assert isinstance(STT_CHUNK_DURATION_SECONDS, int)
        assert STT_CHUNK_DURATION_SECONDS > 0

    def test_chunk_duration_under_api_limit(self):
        """Google STT V2 recognize() hard limit is 60 s per request."""
        from config.voice import STT_CHUNK_DURATION_SECONDS
        assert STT_CHUNK_DURATION_SECONDS < 60

    def test_sync_limit_is_positive_int(self):
        from config.voice import STT_SYNC_LIMIT_SECONDS
        assert isinstance(STT_SYNC_LIMIT_SECONDS, int)
        assert STT_SYNC_LIMIT_SECONDS > 0

    def test_sync_limit_exceeds_chunk_duration(self):
        """Sync limit must be > chunk step so short audio isn't falsely chunked."""
        from config.voice import STT_CHUNK_DURATION_SECONDS, STT_SYNC_LIMIT_SECONDS
        assert STT_SYNC_LIMIT_SECONDS > STT_CHUNK_DURATION_SECONDS

    def test_sync_limit_under_api_hard_limit(self):
        """V2 sync hard limit is 60 s — our configured limit must stay below it."""
        from config.voice import STT_SYNC_LIMIT_SECONDS
        assert STT_SYNC_LIMIT_SECONDS < 60

    def test_max_audio_size_mb_positive(self):
        from config.voice import MAX_AUDIO_SIZE_MB
        assert isinstance(MAX_AUDIO_SIZE_MB, int)
        assert MAX_AUDIO_SIZE_MB > 0

    def test_max_audio_size_bytes_derived_correctly(self):
        from config.voice import MAX_AUDIO_SIZE_BYTES, MAX_AUDIO_SIZE_MB
        assert MAX_AUDIO_SIZE_BYTES == MAX_AUDIO_SIZE_MB * 1024 * 1024

    def test_max_audio_duration_positive(self):
        from config.voice import MAX_AUDIO_DURATION_SECONDS
        assert isinstance(MAX_AUDIO_DURATION_SECONDS, int)
        assert MAX_AUDIO_DURATION_SECONDS > 0


# ---------------------------------------------------------------------------
# _resolve_project_id — resolution priority: yaml → env → credentials file
# ---------------------------------------------------------------------------

class TestResolveProjectId:
    """Tests for config.voice._resolve_project_id (called at import time).

    We call it directly rather than relying on the module-level
    STT_PROJECT_ID constant, which is frozen at first import.
    """

    def _resolve(self, yaml_val="", env_val="", cred_file_project=""):
        """Drive _resolve_project_id with controlled inputs."""
        import config.voice as cv

        # Temporarily replace the module-level _stt dict's project_id
        original_stt = cv._stt

        patched_stt = dict(original_stt)
        patched_stt["project_id"] = yaml_val

        env_overrides = {}
        if env_val:
            env_overrides["JA_VOICE_GOOGLE_STT_PROJECT_ID"] = env_val
        else:
            env_overrides.pop("JA_VOICE_GOOGLE_STT_PROJECT_ID", None)

        with patch.object(cv, "_stt", patched_stt):
            with patch.dict(os.environ, env_overrides, clear=False):
                if cred_file_project:
                    cred_data = {"project_id": cred_file_project}
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as f:
                        json.dump(cred_data, f)
                        cred_path = f.name
                    try:
                        with patch.dict(
                            os.environ,
                            {"GOOGLE_APPLICATION_CREDENTIALS": cred_path},
                        ):
                            return cv._resolve_project_id()
                    finally:
                        os.unlink(cred_path)
                else:
                    with patch.dict(
                        os.environ,
                        {"GOOGLE_APPLICATION_CREDENTIALS": ""},
                    ):
                        return cv._resolve_project_id()

    def test_yaml_value_returned_first(self):
        result = self._resolve(
            yaml_val="from-yaml",
            env_val="from-env",
            cred_file_project="from-cred",
        )
        assert result == "from-yaml"

    def test_env_var_returned_when_yaml_empty(self):
        result = self._resolve(
            yaml_val="",
            env_val="from-env",
            cred_file_project="from-cred",
        )
        assert result == "from-env"

    def test_credentials_file_used_as_last_resort(self):
        result = self._resolve(
            yaml_val="",
            env_val="",
            cred_file_project="from-cred-file",
        )
        assert result == "from-cred-file"

    def test_all_empty_returns_empty_string(self):
        result = self._resolve(yaml_val="", env_val="", cred_file_project="")
        assert result == ""

    def test_yaml_whitespace_only_treated_as_empty(self):
        result = self._resolve(yaml_val="   ", env_val="from-env")
        assert result == "from-env"

    def test_env_whitespace_only_treated_as_empty(self):
        result = self._resolve(
            yaml_val="",
            env_val="   ",
            cred_file_project="from-cred-file",
        )
        assert result == "from-cred-file"

    def test_missing_project_id_in_cred_file_falls_through(self):
        """Credentials file without project_id key → return empty string."""
        import config.voice as cv
        import tempfile

        cred_data = {"type": "service_account", "client_email": "x@y.com"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(cred_data, f)
            cred_path = f.name

        try:
            patched_stt = dict(cv._stt)
            patched_stt["project_id"] = ""
            with patch.object(cv, "_stt", patched_stt):
                with patch.dict(
                    os.environ,
                    {
                        "GOOGLE_APPLICATION_CREDENTIALS": cred_path,
                        "JA_VOICE_GOOGLE_STT_PROJECT_ID": "",
                    },
                ):
                    result = cv._resolve_project_id()
        finally:
            os.unlink(cred_path)

        assert result == ""

    def test_malformed_cred_file_does_not_raise(self):
        """A non-JSON credentials file must be handled gracefully."""
        import config.voice as cv

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("NOT JSON {{{{")
            bad_path = f.name

        try:
            patched_stt = dict(cv._stt)
            patched_stt["project_id"] = ""
            with patch.object(cv, "_stt", patched_stt):
                with patch.dict(
                    os.environ,
                    {
                        "GOOGLE_APPLICATION_CREDENTIALS": bad_path,
                        "JA_VOICE_GOOGLE_STT_PROJECT_ID": "",
                    },
                ):
                    result = cv._resolve_project_id()  # must not raise
        finally:
            os.unlink(bad_path)

        assert result == ""

    def test_nonexistent_cred_file_returns_empty(self):
        import config.voice as cv

        patched_stt = dict(cv._stt)
        patched_stt["project_id"] = ""
        with patch.object(cv, "_stt", patched_stt):
            with patch.dict(
                os.environ,
                {
                    "GOOGLE_APPLICATION_CREDENTIALS": "/does/not/exist.json",
                    "JA_VOICE_GOOGLE_STT_PROJECT_ID": "",
                },
            ):
                result = cv._resolve_project_id()

        assert result == ""
