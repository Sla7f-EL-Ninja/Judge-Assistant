"""
main.py

Entry point for the Summarization pipeline.

Usage:
    # With document files:
    python main.py doc1.txt doc2.txt

    # With no arguments: runs sample data
    python main.py

The pipeline processes legal case documents through Nodes 0-5 using LangGraph
and produces a judge-facing case brief in formal legal Arabic.
"""

import json
import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import get_llm

# Ensure Summerize directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import create_pipeline


# Load environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# Sample documents for testing



def main():
    """Run the full summarization pipeline."""
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nWARNING: GROQ_API_KEY not found in environment variables.")
        print("Ensure you have a .env file or set the variable.")
        print("The pipeline requires an LLM API key to function.\n")

    # Initialize LLM
    try:
        llm = get_llm("high")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    # Build the pipeline
    app = create_pipeline(llm)

    # Prepare input documents
    if len(sys.argv) > 1:
        # Load documents from file paths
        documents = []
        for file_path in sys.argv[1:]:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}, skipping.")
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                doc_id = os.path.basename(file_path)
                documents.append({"doc_id": doc_id, "raw_text": raw_text})
                print(f"Loaded: {file_path} ({len(raw_text)} chars)")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if not documents:
            print("No valid documents loaded. Exiting.")
            return
    else:
        print(f"No document paths provided.")

    # Run the pipeline
    print("\n" + "=" * 60)
    print("STARTING SUMMARIZATION PIPELINE")
    print("=" * 60)

    initial_state = {
        "documents": documents,
        "chunks": [],
        "classified_chunks": [],
        "bullets": [],
        "role_aggregations": [],
        "themed_roles": [],
        "role_theme_summaries": [],
        "case_brief": {},
        "all_sources": [],
        "rendered_brief": "",
    }

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Output results
    rendered_brief = final_state.get("rendered_brief", "")
    all_sources = final_state.get("all_sources", [])
    case_brief = final_state.get("case_brief", {})

    if rendered_brief:
        print("\n" + "=" * 60)
        print("FINAL CASE BRIEF")
        print("=" * 60)
        print(rendered_brief)
    else:
        print("\nNo brief was generated.")

    print(f"\nTotal unique sources: {len(all_sources)}")

    # Save outputs
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Save full pipeline state as JSON
    output_json_path = os.path.join(output_dir, "pipeline_output.json")
    try:
        serializable_state = {
            "case_brief": case_brief,
            "all_sources": all_sources,
            "rendered_brief": rendered_brief,
            "role_theme_summaries": final_state.get("role_theme_summaries", []),
            "themed_roles": final_state.get("themed_roles", []),
            "role_aggregations": final_state.get("role_aggregations", []),
            "bullets_count": len(final_state.get("bullets", [])),
            "chunks_count": len(final_state.get("chunks", [])),
            "documents_count": len(documents),
        }
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, ensure_ascii=False, indent=2)
        print(f"\nPipeline output saved to: {output_json_path}")
    except Exception as e:
        print(f"Warning: Could not save pipeline output: {e}")

    # Save rendered brief as markdown
    brief_md_path = os.path.join(output_dir, "case_brief.md")
    try:
        with open(brief_md_path, "w", encoding="utf-8") as f:
            f.write(rendered_brief)
        print(f"Case brief saved to: {brief_md_path}")
    except Exception as e:
        print(f"Warning: Could not save case brief: {e}")


if __name__ == "__main__":
    main()
