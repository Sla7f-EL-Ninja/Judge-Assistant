"""
ocr_adapter.py

Adapter for the OCR pipeline (OCR/ocr_pipeline.py).

Wraps ``process_document()`` and returns an AgentResult with the
extracted raw text.
"""

import logging
import sys
import os
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


class OCRAdapter(AgentAdapter):
    """Thin wrapper around the OCR pipeline's ``process_document``."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Run OCR on the uploaded files and return extracted text.

        Parameters
        ----------
        query:
            The judge query (informational only for OCR).
        context:
            Must contain ``uploaded_files`` -- a list of file paths to process.
            Optionally contains ``case_id`` used as ``doc_id``.
        """
        uploaded_files = context.get("uploaded_files", [])
        if not uploaded_files:
            return AgentResult(
                response="",
                error="No files provided for OCR processing.",
            )

        try:
            # Add OCR directory to path for its internal imports
            ocr_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "OCR"
            )
            ocr_dir = os.path.normpath(ocr_dir)
            if ocr_dir not in sys.path:
                sys.path.insert(0, ocr_dir)

            from ocr_pipeline import process_document

            all_texts = []
            for file_path in uploaded_files:
                doc_id = context.get("case_id", None)
                result = process_document(
                    file_path=file_path,
                    doc_id=doc_id,
                )
                all_texts.append(result.raw_text)

            combined = "\n\n---\n\n".join(all_texts)
            return AgentResult(
                response=combined,
                sources=[f"OCR: {fp}" for fp in uploaded_files],
                raw_output={"raw_texts": all_texts},
            )

        except Exception as exc:
            error_msg = f"OCR adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
