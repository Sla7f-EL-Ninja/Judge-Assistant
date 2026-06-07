"""
Voice STT pipeline CLI entry point.

Usage:
    python -m Voice.main <audio_file> [--json] [--output OUTPUT]

Examples:
    python -m Voice.main recording.webm
    python -m Voice.main recording.mp3 --json
    python -m Voice.main recording.wav --json -o result.json

Supported formats:
    Any format Google STT accepts — WebM, MP3, WAV, OGG, MP4, FLAC.
    WebM/Opus is what browsers produce natively via MediaRecorder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe an Arabic audio file using Google STT Chirp 2.",
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file (WebM, WAV, MP3, OGG, FLAC, MP4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Print full JSON result instead of the summary",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write JSON result to this file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return 1

    # ---- Size check (mirrors API limit) ------------------------------------
    from config.voice import MAX_AUDIO_SIZE_MB, MAX_AUDIO_SIZE_BYTES  # noqa
    size_bytes = audio_path.stat().st_size
    if size_bytes > MAX_AUDIO_SIZE_BYTES:
        print(
            f"Error: file is {size_bytes / 1024 / 1024:.1f} MB — "
            f"exceeds the {MAX_AUDIO_SIZE_MB} MB limit.",
            file=sys.stderr,
        )
        return 1

    # ---- Read audio -------------------------------------------------------
    audio_bytes = audio_path.read_bytes()
    print(f"Audio file : {audio_path.name} ({len(audio_bytes) / 1024:.1f} KB)")

    # ---- Transcribe -------------------------------------------------------
    from Voice.stt_engine import transcribe_audio  # noqa
    print("Transcribing …\n")
    result = transcribe_audio(audio_bytes)

    # ---- Output -----------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Result written to {output_path}")

    elif args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        _print_summary(audio_path.name, result)

    return 0 if not result.get("error") else 1


def _print_summary(filename: str, result: dict) -> None:
    print("=" * 60)
    print(f"  File       : {filename}")

    if result.get("error"):
        print(f"  ERROR      : {result['error']}")
        print("=" * 60)
        return

    conf = result.get("confidence")
    conf_str = f"{conf:.4f}" if conf is not None else "N/A"
    transcript = result.get("transcript", "")

    print(f"  Confidence : {conf_str}")
    print(f"  Length     : {len(transcript)} chars")
    print("=" * 60)
    print()
    print("Transcript:")
    print("-" * 60)
    print(transcript if transcript else "(empty — silence detected)")
    print()


if __name__ == "__main__":
    sys.exit(main())
