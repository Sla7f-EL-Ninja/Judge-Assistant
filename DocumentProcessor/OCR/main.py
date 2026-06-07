"""
DocumentProcessor OCR pipeline CLI entry point.

Usage:
    python -m DocumentProcessor.OCR.main <file_path> [--doc-id DOC_ID] [--output OUTPUT] [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from DocumentProcessor.OCR.ocr_pipeline import run_ocr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the OCR pipeline on a document.",
    )
    parser.add_argument("file_path", help="Path to the input file (PDF or image)")
    parser.add_argument("--doc-id", default=None, help="Optional document identifier")
    parser.add_argument("--output", "-o", default=None, help="Write JSON result to this file")
    parser.add_argument("--json", action="store_true", default=False, dest="json_output")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,   # removes any handlers already added by imported modules,
                      # preventing the duplicate-log issue seen in earlier runs
    )

    file_path = args.file_path
    if not Path(file_path).exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    result = run_ocr(file_path=file_path, doc_id=args.doc_id)
    result_dict = result.model_dump()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Result written to {output_path}")
    elif args.json_output:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    else:
        _print_summary(result)

    return 0


def _print_summary(result) -> None:
    meta = result.metadata
    print("=" * 60)
    print(f"  File:      {meta.get('filename', 'unknown')}")
    print(f"  Pages:     {meta.get('total_pages', 0)}")
    print(f"  DPI:       {meta.get('pdf_dpi', '?')}")
    print(f"  Time:      {meta.get('processing_time_seconds', '?')}s")
    print(f"  GCV:       {meta.get('gcv_batch_seconds', '?')}s")
    print(f"  Refined:   {meta.get('pages_refined', 0)} page(s)")
    print(f"  Skipped:   {meta.get('pages_skipped_refine', 0)} page(s)")
    print("=" * 60)
    for page in result.pages:
        print(f"\n--- Page {page.page_number} ---")
        if page.error:
            print(f"  ERROR: {page.error}")
            continue
        conf_str = f"{page.confidence:.4f}" if page.confidence is not None else "N/A"
        print(f"  Confidence:  {conf_str}")
        print(f"  Text length: {len(page.normalized_text)} chars")
        preview = page.normalized_text[:200]
        if preview:
            print(f"  Preview:     {preview}…")
    print()


if __name__ == "__main__":
    sys.exit(main())