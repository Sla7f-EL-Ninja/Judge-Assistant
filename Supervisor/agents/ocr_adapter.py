"""
ocr_adapter.py

Adapter for the OCR pipeline (OCR/ocr_pipeline.py).

Wraps ``process_document()`` and returns an AgentResult with the
extracted raw text.

Performance fix
---------------
The original implementation added the OCR directory to sys.path on every
call inside invoke().  Although it had an ``if ocr_dir not in sys.path``
guard (so the path itself was not duplicated), it still re-executed the
path-setup code and re-imported ``process_document`` from scratch on every
invocation.  The import is cached at the class level so the filesystem
lookup and module initialisation happen only once.
"""

import logging
import os
import sys
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


def _load_process_document():
    """Import and return ``process_document`` from the OCR pipeline.

    Called once; result cached on the class.
    """
    ocr_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "OCR"
    ))
    if ocr_dir not in sys.path:
        sys.path.insert(0, ocr_dir)

    from ocr_pipeline import process_document  # noqa: E402
    return process_document


class OCRAdapter(AgentAdapter):
    """Thin wrapper around the OCR pipeline's ``process_document``."""

    _process_document = None

    @classmethod
    def _get_process_document(cls):
        if cls._process_document is None:
            logger.info("Loading OCR pipeline (first call)...")
            cls._process_document = _load_process_document()
            logger.info("OCR pipeline loaded and cached.")
        return cls._process_document

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
            process_document = self._get_process_document()

            all_texts = []
            for file_path in uploaded_files:
                doc_id = context.get("case_id", None)
                result = process_document(
                    file_path=file_path,
                    doc_id=doc_id,
                )
                all_texts.append(result.raw_text)

            combined = "\n\n---\n\n".join(all_texts)

            # Cap per-text to 50 KB to prevent LangGraph state/checkpoint bloat (P1.3.1)
            _MAX_TEXT_BYTES = 50_000
            capped_texts = [t[:_MAX_TEXT_BYTES] for t in all_texts]
            if any(len(t) > _MAX_TEXT_BYTES for t in all_texts):
                logger.warning(
                    "OCR text capped to %d chars per file to limit state size", _MAX_TEXT_BYTES
                )

            return AgentResult(
                response=combined[:_MAX_TEXT_BYTES * len(uploaded_files)],
                sources=[f"OCR: {fp}" for fp in uploaded_files],
                raw_output={"raw_texts": capped_texts},
            )

        except ImportError as exc:
            error_msg = f"OCR adapter import error: {exc}"
            logger.exception(error_msg)
            # Only reset on import failure — bad input/pipeline error should NOT poison cache
            OCRAdapter._process_document = None
            return AgentResult(response="", error=error_msg)

        except Exception as exc:
            error_msg = f"OCR adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)