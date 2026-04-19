"""
CLI wrapper around summarize.pipeline.run_summarization.

Usage:
    python -m summarize doc1.txt doc2.txt
    python -m summarize doc1.txt doc2.txt --case-id <case_id>
"""

import argparse
import json
import os

from dotenv import load_dotenv

from summarize.pipeline import run_summarization

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hakim summarization pipeline on one or more legal documents."
    )
    parser.add_argument("files", nargs="*", metavar="FILE")
    parser.add_argument(
        "--case-id",
        default=None,
        help="Case ID to link the summary to in MongoDB (settings.yaml handles connection).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.files:
        print("No document paths provided.")
        return

    documents = []
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}, skipping.")
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            documents.append({"doc_id": os.path.basename(file_path), "raw_text": raw_text})
            print(f"Loaded: {file_path} ({len(raw_text)} chars)")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not documents:
        print("No valid documents loaded. Exiting.")
        return

    print("\n" + "=" * 60)
    print("STARTING SUMMARIZATION PIPELINE")
    print("=" * 60)

    try:
        result = run_summarization(documents=documents, case_id=args.case_id)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if result.rendered_brief:
        print("\n" + "=" * 60)
        print("FINAL CASE BRIEF")
        print("=" * 60)
        print(result.rendered_brief)
    else:
        print("\nNo brief was generated.")

    print(f"\nTotal unique sources: {len(result.all_sources)}")

    if args.case_id:
        status = "saved" if result.saved_to_db else "FAILED to save"
        print(f"MongoDB: {status}  (case_id='{args.case_id}')")

    output_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(output_dir, "pipeline_output.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "case_brief":           result.case_brief,
                "all_sources":          result.all_sources,
                "rendered_brief":       result.rendered_brief,
                "role_theme_summaries": result.role_theme_summaries,
                "themed_roles":         result.themed_roles,
                "role_aggregations":    result.role_aggregations,
                "bullets_count":        result.bullets_count,
                "chunks_count":         result.chunks_count,
                "documents_count":      result.documents_count,
            }, f, ensure_ascii=False, indent=2)
        print(f"Pipeline output saved to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save pipeline output: {e}")

    md_path = os.path.join(output_dir, "case_brief.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.rendered_brief)
        print(f"Case brief saved to: {md_path}")
    except Exception as e:
        print(f"Warning: Could not save case brief: {e}")


if __name__ == "__main__":
    main()
