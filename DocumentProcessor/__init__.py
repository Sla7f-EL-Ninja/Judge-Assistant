"""DocumentProcessor — standalone document ingestion pipeline."""

from DocumentProcessor.pipeline import process_document, process_document_group, reindex_document

__all__ = ["process_document", "process_document_group", "reindex_document"]
