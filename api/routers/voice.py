"""
api/routers/voice.py
---------------------
Voice transcription endpoint.

  POST /api/v1/voice/transcribe

Accepts a browser audio recording (WebM/Opus from MediaRecorder),
transcribes it with Google STT Chirp 2, and returns the Arabic text
ready to drop into the chat input field.

The transcript goes directly to the supervisor — no LLM normalisation.
Nothing is stored; the endpoint is stateless.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from api.dependencies import get_current_user
from api.schemas.common import ErrorEnvelope
from api.schemas.voice import TranscriptionResponse
from config.voice import MAX_AUDIO_SIZE_BYTES, MAX_AUDIO_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])

# Accepted MIME types from browser MediaRecorder
_ALLOWED_MIME_PREFIXES = (
    "audio/webm",
    "audio/ogg",
    "audio/wav",
    "audio/mp4",
    "audio/mpeg",
    "audio/",   # catch-all for any audio format GCV can handle
)


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe Arabic voice input to text",
    description=(
        "Upload an audio recording and receive the Arabic transcription. "
        "Supports Modern Standard Arabic and Egyptian colloquial. "
        "The returned transcript can be placed directly into the chat input."
    ),
    responses={
        400: {"model": ErrorEnvelope, "description": "Audio too large or invalid"},
        401: {"model": ErrorEnvelope},
        500: {"model": ErrorEnvelope, "description": "STT engine error"},
    },
)
async def transcribe_voice(
    audio: UploadFile = File(
        ...,
        description="Audio file from browser MediaRecorder (WebM/Opus preferred)",
    ),
    user_id: str = Depends(get_current_user),
) -> TranscriptionResponse:
    """Transcribe an audio file to Arabic text."""

    # ---- Validate content type -------------------------------------------
    content_type = (audio.content_type or "").lower()
    if content_type and not any(
        content_type.startswith(p) for p in _ALLOWED_MIME_PREFIXES
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_AUDIO_FORMAT",
                "message": (
                    f"Unsupported audio format '{content_type}'. "
                    "Send audio/webm (MediaRecorder default)."
                ),
            },
        )

    # ---- Read and validate size ------------------------------------------
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_AUDIO", "message": "Audio file is empty."},
        )

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "AUDIO_TOO_LARGE",
                "message": (
                    f"Audio exceeds the {MAX_AUDIO_SIZE_MB} MB limit. "
                    "Please keep recordings under 2 minutes."
                ),
            },
        )

    # ---- Transcribe ------------------------------------------------------
    logger.info(
        "Transcribing audio: user=%s size=%.1f KB type=%s",
        user_id,
        len(audio_bytes) / 1024,
        content_type or "unknown",
    )

    try:
        from Voice.stt_engine import transcribe_audio  # noqa: PLC0415
        result = transcribe_audio(audio_bytes)
    except RuntimeError as exc:
        # Raised by get_stt_engine() when project_id is missing
        logger.error("STT engine configuration error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"code": "STT_CONFIG_ERROR", "message": str(exc)},
        )
    except Exception as exc:
        logger.exception("Unexpected STT error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"code": "STT_ERROR", "message": "Transcription failed."},
        )

    if result.get("error"):
        logger.warning("STT returned error: %s", result["error"])
        raise HTTPException(
            status_code=500,
            detail={"code": "STT_ERROR", "message": result["error"]},
        )

    logger.info(
        "Transcription complete: %d chars, confidence=%s",
        len(result.get("transcript", "")),
        result.get("confidence"),
    )

    return TranscriptionResponse(
        transcript=result.get("transcript", ""),
        confidence=result.get("confidence"),
        error=None,
    )