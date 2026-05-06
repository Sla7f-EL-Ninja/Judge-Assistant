"""DocumentProcessor — standalone document ingestion pipeline."""

from DocumentProcessor.pipeline import process_document, reindex_document

__all__ = ["process_document", "reindex_document"]
