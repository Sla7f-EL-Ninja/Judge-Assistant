# """
# config.voice
# ------------
# Voice / STT pipeline constants sourced from settings.yaml.

# The GCP project ID is required for the Chirp 2 V2 recognizer path.
# It can be set in three ways (checked in order):
#   1. settings.yaml  voice.google_stt.project_id
#   2. .env           JA_VOICE_GOOGLE_STT_PROJECT_ID=your-project-id
#   3. Extracted automatically from the service-account JSON file
#      pointed to by GOOGLE_APPLICATION_CREDENTIALS
# """

# from __future__ import annotations

# import json
# import logging
# import os

# from config import cfg

# logger = logging.getLogger(__name__)

# _voice = cfg.get("voice", {})
# _stt = _voice.get("google_stt", {})


# def _resolve_project_id() -> str:
#     """Return GCP project ID from config, env, or credentials file."""
#     # 1. Explicit setting
#     from_yaml = _stt.get("project_id", "").strip()
#     if from_yaml:
#         return from_yaml

#     # 2. Direct env var (also set by JA_ override convention)
#     from_env = os.getenv("JA_VOICE_GOOGLE_STT_PROJECT_ID", "").strip()
#     if from_env:
#         return from_env

#     # 3. Extract from credentials JSON file
#     cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
#     if cred_file and os.path.exists(cred_file):
#         try:
#             with open(cred_file, "r", encoding="utf-8") as f:
#                 creds = json.load(f)
#             project_id = creds.get("project_id", "").strip()
#             if project_id:
#                 logger.info(
#                     "STT project_id resolved from credentials file: %s",
#                     project_id,
#                 )
#                 return project_id
#         except Exception as exc:
#             logger.warning(
#                 "Could not read project_id from credentials file: %s", exc
#             )

#     return ""


# # -----------------------------
# # Google STT
# # -----------------------------
# STT_MODEL: str = _stt.get("model", "chirp_2")
# STT_LANGUAGE_CODES: list = _stt.get("language_codes", ["ar-EG"])
# STT_PROJECT_ID: str = _resolve_project_id()

# # -----------------------------
# # Limits
# # -----------------------------
# MAX_AUDIO_SIZE_MB: int = _voice.get("max_audio_size_mb", 10)
# MAX_AUDIO_SIZE_BYTES: int = MAX_AUDIO_SIZE_MB * 1024 * 1024
# MAX_AUDIO_DURATION_SECONDS: int = _voice.get("max_audio_duration_seconds", 120)



"""
config.voice
------------
Voice / STT pipeline constants sourced from settings.yaml.
"""

from __future__ import annotations

import json
import logging
import os

from config import cfg

logger = logging.getLogger(__name__)

_voice = cfg.get("voice", {})
_stt = _voice.get("google_stt", {})


def _resolve_project_id() -> str:
    """Return GCP project ID from config, env, or credentials file."""
    from_yaml = _stt.get("project_id", "").strip()
    if from_yaml:
        return from_yaml

    from_env = os.getenv("JA_VOICE_GOOGLE_STT_PROJECT_ID", "").strip()
    if from_env:
        return from_env

    cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_file and os.path.exists(cred_file):
        try:
            with open(cred_file, "r", encoding="utf-8") as f:
                creds = json.load(f)
            project_id = creds.get("project_id", "").strip()
            if project_id:
                logger.info(
                    "STT project_id resolved from credentials file: %s",
                    project_id,
                )
                return project_id
        except Exception as exc:
            logger.warning(
                "Could not read project_id from credentials file: %s", exc
            )

    return ""


# -----------------------------
# Google STT
# -----------------------------
STT_MODEL: str = _stt.get("model", "chirp_2")
STT_LANGUAGE_CODES: list = _stt.get("language_codes", ["ar-EG"])
STT_PROJECT_ID: str = _resolve_project_id()
STT_LOCATION: str = _stt.get("location", "europe-west4")

# -----------------------------
# Chunking / parallelism
# -----------------------------
_chunking = _voice.get("chunking", {})
STT_MAX_WORKERS: int = _chunking.get("max_workers", 4)
STT_CHUNK_DURATION_SECONDS: int = _chunking.get("chunk_duration_seconds", 50)
STT_SYNC_LIMIT_SECONDS: int = _chunking.get("sync_limit_seconds", 55)

# -----------------------------
# Limits
# -----------------------------
MAX_AUDIO_SIZE_MB: int = _voice.get("max_audio_size_mb", 10)
MAX_AUDIO_SIZE_BYTES: int = MAX_AUDIO_SIZE_MB * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS: int = _voice.get("max_audio_duration_seconds", 120)