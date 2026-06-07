"""
Voice — Arabic speech-to-text input layer.

Public API
----------
    from Voice.stt_engine import get_stt_engine, transcribe_audio
"""

from Voice.stt_engine import get_stt_engine, transcribe_audio

__all__ = ["get_stt_engine", "transcribe_audio"]