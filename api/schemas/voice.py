"""
api/schemas/voice.py
---------------------
Pydantic schemas for the voice transcription endpoint.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Response from POST /api/v1/voice/transcribe."""

    transcript: str = Field(
        description=(
            "Transcribed Arabic text ready to paste into the chat input. "
            "Empty string when the audio contained only silence."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Mean word confidence from the STT engine (0.0–1.0). "
                    "None when the engine did not return confidence scores.",
        ge=0.0,
        le=1.0,
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if transcription failed. "
                    "Non-null only when the HTTP status is also non-200.",
    )