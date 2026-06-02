# """
# DocumentProcessor.pipeline
# ---------------------------
# Unified document processing: ingest → OCR/extract → classify → store.

# Public API
# ----------
#     process_document(file_path, case_id, file_id)       -> dict
#     process_document_group(file_paths, case_id, file_ids) -> dict
#     reindex_document(mongo_id, new_text, doc_meta)      -> int

# OCR changes (GCV migration)
# ----------------------------
# ``_extract_text_via_ocr`` now returns an ``_OCRExtraction`` dataclass
# instead of a bare string.  It carries:

#     text            — canonical text (LLM-refined if refinement enabled,
#                       otherwise normalized OCR text).  This is what gets
#                       classified and indexed.
#     raw_ocr_text    — verbatim GCV output, stored in MongoDB for audit.
#                       NOT exposed in the API.
#     word_confidences — list[dict] ready for MongoDB storage and API responses.

# Word confidence dicts follow the schema::

#     {"word": str, "confidence": float, "band": "high"|"mid"|"low",
#      "page_number": int}
# """

# from __future__ import annotations

# import logging
# import os
# import re
# import threading
# from dataclasses import dataclass, field
# from datetime import datetime, timezone
# from typing import Any, Dict, List, Optional

# from pymongo import MongoClient

# from config.supervisor import (
#     EMBEDDING_MODEL,
#     MONGO_COLLECTION,
#     MONGO_DB,
#     MONGO_URI,
#     QDRANT_COLLECTION_CASE,
#     QDRANT_GRPC_PORT,
#     QDRANT_HOST,
#     QDRANT_PORT,
#     QDRANT_PREFER_GRPC,
# )
# from DocumentProcessor.classifier import classify_document
# from DocumentProcessor.OCR.ocr_pipeline import run_ocr

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # File type constants
# # ---------------------------------------------------------------------------

# TEXT_EXTENSIONS = {".txt", ".text", ".csv", ".json", ".md"}
# PDF_EXTENSIONS = {".pdf"}
# IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# _MAGIC_BYTES = {
#     b"%PDF": "pdf",
#     b"\x89PNG": "image",
#     b"\xff\xd8\xff": "image",
#     b"GIF8": "image",
#     b"BM": "image",
#     b"RIFF": "image",
#     b"\x49\x49\x2a\x00": "image",
#     b"\x4d\x4d\x00\x2a": "image",
# }

# PAGE_SEP = "\n\n--- PAGE BREAK ---\n\n"

# # ---------------------------------------------------------------------------
# # OCR extraction result
# # ---------------------------------------------------------------------------


# @dataclass
# class _OCRExtraction:
#     """Structured result from the OCR + LLM refinement pipeline."""

#     text: str                                  # canonical (refined or normalized)
#     raw_ocr_text: str = ""                     # verbatim GCV output (audit only)
#     word_confidences: List[Dict] = field(default_factory=list)


# # ---------------------------------------------------------------------------
# # Lazy singletons — double-checked locking
# # ---------------------------------------------------------------------------

# _mongo_client: Optional[MongoClient] = None
# _mongo_lock = threading.Lock()

# _vectorstore = None
# _vectorstore_lock = threading.Lock()

# _minio_config: Dict[str, Any] = {
#     "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
#     "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
#     "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
#     "bucket": os.getenv("MINIO_BUCKET", "hakim-files"),
#     "secure": False,
# }


# def _get_mongo_collection():
#     global _mongo_client
#     if _mongo_client is None:
#         with _mongo_lock:
#             if _mongo_client is None:
#                 _mongo_client = MongoClient(MONGO_URI)
#     return _mongo_client[MONGO_DB][MONGO_COLLECTION]


# def _get_vectorstore():
#     global _vectorstore
#     if _vectorstore is None:
#         with _vectorstore_lock:
#             if _vectorstore is None:
#                 from langchain_huggingface import HuggingFaceEmbeddings
#                 from langchain_qdrant import QdrantVectorStore
#                 from qdrant_client import QdrantClient
#                 from qdrant_client.models import Distance, VectorParams

#                 embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#                 client = QdrantClient(
#                     host=QDRANT_HOST,
#                     port=QDRANT_PORT,
#                     grpc_port=QDRANT_GRPC_PORT,
#                     prefer_grpc=QDRANT_PREFER_GRPC,
#                 )

#                 existing = [c.name for c in client.get_collections().collections]
#                 if QDRANT_COLLECTION_CASE not in existing:
#                     try:
#                         client.create_collection(
#                             collection_name=QDRANT_COLLECTION_CASE,
#                             vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
#                         )
#                         logger.info("Created Qdrant collection '%s'", QDRANT_COLLECTION_CASE)
#                     except Exception as exc:
#                         logger.info(
#                             "Collection '%s' already exists (concurrent create): %s",
#                             QDRANT_COLLECTION_CASE, exc,
#                         )

#                 _vectorstore = QdrantVectorStore(
#                     client=client,
#                     collection_name=QDRANT_COLLECTION_CASE,
#                     embedding=embeddings,
#                 )
#     return _vectorstore


# # ---------------------------------------------------------------------------
# # File type detection
# # ---------------------------------------------------------------------------

# def detect_file_type(file_path: str) -> str:
#     """Return ``'text'``, ``'pdf'``, ``'image'``, or ``'unknown'``."""
#     try:
#         with open(file_path, "rb") as f:
#             header = f.read(8)
#         for magic, ftype in _MAGIC_BYTES.items():
#             if header.startswith(magic):
#                 return ftype
#     except OSError:
#         pass

#     _, ext = os.path.splitext(file_path)
#     ext = ext.lower()
#     if ext in TEXT_EXTENSIONS:
#         return "text"
#     if ext in PDF_EXTENSIONS:
#         return "pdf"
#     if ext in IMAGE_EXTENSIONS:
#         return "image"
#     return "unknown"


# # ---------------------------------------------------------------------------
# # Text extraction
# # ---------------------------------------------------------------------------

# def _extract_text_from_file(file_path: str) -> str:
#     for encoding in ("utf-8", "cp1256", "windows-1252"):
#         try:
#             with open(file_path, "r", encoding=encoding) as f:
#                 return f.read()
#         except UnicodeDecodeError:
#             continue
#     with open(file_path, "r", encoding="utf-8", errors="replace") as f:
#         logger.warning("File '%s' decoded with replacement chars", file_path)
#         return f.read()


# def _extract_text_from_pdf(file_path: str) -> str:
#     try:
#         from pypdf import PdfReader
#     except ImportError:
#         logger.error("pypdf required for PDF extraction. pip install pypdf")
#         return ""

#     try:
#         reader = PdfReader(file_path)
#         pages_text = [
#             page.extract_text().strip()
#             for page in reader.pages
#             if page.extract_text()
#         ]
#         result = "\n\n".join(pages_text)
#         if not result:
#             logger.warning(
#                 "PDF '%s' extracted to empty text — may be a scanned/image PDF",
#                 file_path,
#             )
#         return result
#     except Exception as exc:
#         logger.exception("Failed to extract text from PDF '%s': %s", file_path, exc)
#         raise RuntimeError(f"PDF extraction failed: {exc}") from exc


# def _extract_text_via_ocr(
#     file_path: str,
#     doc_id: Optional[str] = None,
# ) -> _OCRExtraction:
#     """Run the OCR + LLM pipeline and return an ``_OCRExtraction``.

#     The returned ``text`` field is the LLM-refined text (or normalized OCR
#     text when refinement is disabled / fails).  ``raw_ocr_text`` is the
#     verbatim GCV output stored for audit purposes.  ``word_confidences``
#     is a list of serialisable dicts suitable for MongoDB storage.
#     """
#     try:
#         result = run_ocr(file_path=file_path, doc_id=doc_id)

#         all_word_confidences: List[Dict] = []
#         canonical_pages: List[str] = []
#         raw_pages: List[str] = []

#         for page in result.pages:
#             if not page.raw_text:
#                 continue

#             # Canonical text: prefer refined; fall back to normalized
#             canonical_pages.append(page.canonical_text)
#             raw_pages.append(page.raw_text)

#             if page.word_confidences:
#                 for wc in page.word_confidences:
#                     all_word_confidences.append(wc.model_dump())

#         return _OCRExtraction(
#             text="\n\n".join(canonical_pages),
#             raw_ocr_text="\n\n".join(raw_pages),
#             word_confidences=all_word_confidences,
#         )

#     except Exception as exc:
#         logger.exception("OCR pipeline failed for '%s': %s", file_path, exc)
#         return _OCRExtraction(text="", raw_ocr_text="", word_confidences=[])


# def _extract_text(
#     file_path: str,
#     file_type: str,
#     case_id: str,
# ) -> _OCRExtraction:
#     """Extract text from any supported file type.

#     Always returns ``_OCRExtraction`` for a uniform caller interface.
#     Non-OCR paths (text, PDF) set ``raw_ocr_text`` and ``word_confidences``
#     to empty values.
#     """
#     if file_type == "text":
#         return _OCRExtraction(text=_extract_text_from_file(file_path))
#     if file_type == "pdf":
#         return _OCRExtraction(text=_extract_text_from_pdf(file_path))
#     if file_type == "image":
#         return _extract_text_via_ocr(file_path, doc_id=case_id)

#     # Unknown — try plain text as a best-effort fallback
#     try:
#         return _OCRExtraction(text=_extract_text_from_file(file_path))
#     except Exception:
#         return _OCRExtraction(text="")


# # ---------------------------------------------------------------------------
# # Storage helpers
# # ---------------------------------------------------------------------------

# def _store_in_mongo(
#     title: str,
#     doc_type: str,
#     case_id: str,
#     source_file: str,
#     text: str,
#     confidence: int,
#     explanation: str,
#     file_type: str,
#     raw_ocr_text: str = "",
#     word_confidences: Optional[List[Dict]] = None,
#     file_id: Optional[str] = None,
#     file_ids: Optional[List[str]] = None,
#     source_files: Optional[List[str]] = None,
# ) -> Optional[Any]:
#     source_file = source_file.replace("\\", "/")
#     doc_record: Dict[str, Any] = {
#         "title": title,
#         "doc_type": doc_type,
#         "case_id": case_id,
#         "source_file": source_file,
# <<<<<<< HEAD
#         "text": text,                               # canonical (refined) text
#         "raw_ocr_text": raw_ocr_text,              # verbatim GCV output
#         "word_confidences": word_confidences or [], # per-word confidence for UI
# =======
#         "text": text,
#         "original_text": text,  # preserved forever; never overwritten by corrections
# >>>>>>> f3e4a2844e5b1434dce0841bc183f94f48c89672
#         "classification_confidence": confidence,
#         "classification_explanation": explanation,
#         "file_type": file_type,
#         "file_id": file_id,
#         "storage_backend": "local",
#         "minio_object": None,
#         "created_at": datetime.now(timezone.utc),
#         "corrected": False,
#     }
#     if file_ids is not None:
#         doc_record["file_ids"] = file_ids
#     if source_files is not None:
#         doc_record["source_files"] = [sf.replace("\\", "/") for sf in source_files]

#     try:
#         col = _get_mongo_collection()

#         # Deduplication only for the legacy single-file path
#         if file_ids is None:
#             existing = col.find_one(
#                 {"source_file": source_file, "case_id": case_id},
#                 {"_id": 1},
#             )
#             if existing:
#                 logger.info(
#                     "Skipping duplicate ingest: source_file='%s' case_id='%s'",
#                     source_file, case_id,
#                 )
#                 return existing["_id"]

#         result = col.insert_one(doc_record)
#         logger.info("Stored in MongoDB: title='%s' id=%s", title, result.inserted_id)
#         return result.inserted_id

#     except Exception as exc:
#         logger.exception("MongoDB insert failed for '%s': %s", title, exc)
#         return None


# def _upload_to_minio(file_path: str, mongo_id: str, case_id: str) -> Optional[str]:
#     try:
#         from minio import Minio

#         cfg = _minio_config
#         client = Minio(
#             endpoint=cfg["endpoint"],
#             access_key=cfg["access_key"],
#             secret_key=cfg["secret_key"],
#             secure=cfg["secure"],
#         )

#         if not client.bucket_exists(cfg["bucket"]):
#             try:
#                 client.make_bucket(cfg["bucket"])
#             except Exception:
#                 pass

#         filename = os.path.basename(file_path)
#         safe_case_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", case_id) if case_id else "no-case"
#         object_name = f"{safe_case_id}/{mongo_id}/{filename}"

#         file_size = os.path.getsize(file_path)
#         with open(file_path, "rb") as f:
#             client.put_object(
#                 bucket_name=cfg["bucket"],
#                 object_name=object_name,
#                 data=f,
#                 length=file_size,
#             )

#         logger.info("Uploaded to MinIO: %s/%s", cfg["bucket"], object_name)
#         return object_name

#     except Exception as exc:
#         logger.warning("MinIO upload failed (non-fatal): %s", exc)
#         return None


# def _index_in_vectorstore(
#     text: str,
#     title: str,
#     doc_type: str,
#     case_id: str,
#     source_file: str,
#     mongo_id: str,
#     file_id: Optional[str] = None,
# ) -> int:
#     """Chunk and index text. Returns the number of chunks indexed."""
#     try:
#         from langchain_text_splitters import RecursiveCharacterTextSplitter
#     except ImportError:
#         from langchain.text_splitters import RecursiveCharacterTextSplitter

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", ".", " ", ""],
#     )

#     chunks = splitter.split_text(text)
#     if not chunks:
#         logger.warning("No chunks produced from text for '%s'", source_file)
#         return 0

#     metadatas = [
#         {
#             "title": title,
#             "type": doc_type,
#             "case_id": case_id,
#             "source_file": source_file,
#             "mongo_id": mongo_id,
#             "file_id": file_id,
#             "chunk_index": i,
#         }
#         for i in range(len(chunks))
#     ]

#     try:
#         _get_vectorstore().add_texts(texts=chunks, metadatas=metadatas)
#         logger.info(
#             "Indexed %d chunk(s) for '%s' (case=%s)", len(chunks), title, case_id
#         )
#         return len(chunks)
#     except Exception as exc:
#         logger.exception(
#             "Vector store indexing failed for '%s': %s", source_file, exc
#         )
#         return 0


# def _delete_qdrant_chunks_by_mongo_id(mongo_id: str) -> None:
#     """Delete all Qdrant points where metadata.mongo_id == mongo_id."""
#     from qdrant_client.models import FieldCondition, Filter, MatchValue

#     vs = _get_vectorstore()
#     client = vs.client
#     filt = Filter(
#         must=[FieldCondition(key="metadata.mongo_id", match=MatchValue(value=mongo_id))]
#     )
#     client.delete(collection_name=vs.collection_name, points_selector=filt)
#     logger.info("Deleted Qdrant chunks for mongo_id=%s", mongo_id)


# # ---------------------------------------------------------------------------
# # Public API — multi-file group (primary ingestion path)
# # ---------------------------------------------------------------------------

# def process_document_group(
#     file_paths: List[str],
#     case_id: str,
#     file_ids: List[str],
# ) -> Dict[str, Any]:
#     """Process a group of files as one multi-page document.

#     Files are processed individually in order; their canonical texts are
#     joined with ``PAGE_SEP``.  Word confidences from all pages are merged
#     into a flat list (each carries a ``page_number`` relative to its OCR run).
#     Classification and Qdrant indexing happen once on the merged text.
#     """
#     per_file_extractions: List[_OCRExtraction] = []
#     file_types: List[str] = []

#     for path in file_paths:
#         ft = detect_file_type(path)
#         file_types.append(ft)
#         extraction = _extract_text(path, ft, case_id)
#         per_file_extractions.append(extraction)

#     merged_text = PAGE_SEP.join(
#         e.text for e in per_file_extractions if e.text
#     ).strip()

#     merged_raw_ocr = PAGE_SEP.join(
#         e.raw_ocr_text for e in per_file_extractions if e.raw_ocr_text
#     ).strip()

#     merged_word_confidences: List[Dict] = [
#         wc
#         for extraction in per_file_extractions
#         for wc in extraction.word_confidences
#     ]

#     unique_types = set(file_types)
#     composite_type = unique_types.pop() if len(unique_types) == 1 else "mixed"

#     source_files = [os.path.basename(p) for p in file_paths]
#     primary_source = source_files[0] if source_files else ""
#     primary_file_id = file_ids[0] if file_ids else None

#     _empty_meta = {
#         "mongo_id": None,
#         "minio_object": None,
#         "qdrant_chunks": 0,
#         "case_id": case_id,
#         "source_file": primary_source,
#         "source_files": source_files,
#         "file_id": primary_file_id,
#         "file_ids": file_ids,
#     }

#     if not merged_text:
#         logger.warning("No text extracted from group %s", file_ids)
#         return {
#             "text": "",
#             "file_type": composite_type,
#             "classification": {
#                 "final_type": "مستند غير معروف",
#                 "confidence": 0,
#                 "explanation": "No text could be extracted",
#             },
#             "metadata": _empty_meta,
#         }

#     classification = classify_document(merged_text)
#     doc_type = classification.get("final_type") or "مستند غير معروف"
#     confidence = classification.get("confidence", 0)
#     explanation = classification.get("explanation", "")

#     mongo_id = _store_in_mongo(
#         title=doc_type,
#         doc_type=doc_type,
#         case_id=case_id,
#         source_file=primary_source,
#         text=merged_text,
#         confidence=confidence,
#         explanation=explanation,
#         file_type=composite_type,
#         raw_ocr_text=merged_raw_ocr,
#         word_confidences=merged_word_confidences,
#         file_id=primary_file_id,
#         file_ids=file_ids,
#         source_files=source_files,
#     )

#     qdrant_chunks = _index_in_vectorstore(
#         text=merged_text,
#         title=doc_type,
#         doc_type=doc_type,
#         case_id=case_id,
#         source_file=primary_source,
#         mongo_id=str(mongo_id) if mongo_id else "",
#         file_id=primary_file_id,
#     )

#     logger.info(
#         "Group %s processed: type='%s' confidence=%d mongo_id=%s chunks=%d words=%d",
#         file_ids, doc_type, confidence, mongo_id, qdrant_chunks,
#         len(merged_word_confidences),
#     )

#     return {
#         "text": merged_text,
#         "file_type": composite_type,
#         "classification": classification,
#         "metadata": {
#             "mongo_id": str(mongo_id) if mongo_id else None,
#             "minio_object": None,
#             "qdrant_chunks": qdrant_chunks,
#             "case_id": case_id,
#             "source_file": primary_source,
#             "source_files": source_files,
#             "file_id": primary_file_id,
#             "file_ids": file_ids,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Public API — single document (legacy path)
# # ---------------------------------------------------------------------------

# def process_document(
#     file_path: str,
#     case_id: str = "",
#     file_id: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """Process a single document end-to-end.

#     Returns
#     -------
#     dict with keys:
#         text           -- canonical extracted text (refined if OCR)
#         file_type      -- 'text' | 'pdf' | 'image' | 'unknown'
#         classification -- {final_type, confidence, explanation}
#         metadata       -- {mongo_id, minio_object, qdrant_chunks, …}
#     """
#     file_type = detect_file_type(file_path)
#     extraction = _extract_text(file_path, file_type, case_id)

#     _empty_meta = {
#         "mongo_id": None,
#         "minio_object": None,
#         "qdrant_chunks": 0,
#         "case_id": case_id,
#         "source_file": file_path,
#         "file_id": file_id,
#     }

#     if not extraction.text or not extraction.text.strip():
#         logger.warning("No text extracted from '%s'", file_path)
#         return {
#             "text": "",
#             "file_type": file_type,
#             "classification": {
#                 "final_type": "مستند غير معروف",
#                 "confidence": 0,
#                 "explanation": "No text could be extracted",
#             },
#             "metadata": _empty_meta,
#         }

#     classification = classify_document(extraction.text)
#     doc_type = classification.get("final_type") or "مستند غير معروف"
#     confidence = classification.get("confidence", 0)
#     explanation = classification.get("explanation", "")

#     mongo_id = _store_in_mongo(
#         title=doc_type,
#         doc_type=doc_type,
#         case_id=case_id,
#         source_file=file_path,
#         text=extraction.text,
#         confidence=confidence,
#         explanation=explanation,
#         file_type=file_type,
#         raw_ocr_text=extraction.raw_ocr_text,
#         word_confidences=extraction.word_confidences,
#         file_id=file_id,
#     )

#     minio_object = None
#     if mongo_id and file_type != "unknown":
#         minio_object = _upload_to_minio(
#             file_path=file_path,
#             mongo_id=str(mongo_id),
#             case_id=case_id,
#         )
#         if minio_object:
#             try:
#                 _get_mongo_collection().update_one(
#                     {"_id": mongo_id},
#                     {"$set": {"minio_object": minio_object, "storage_backend": "minio"}},
#                 )
#             except Exception as exc:
#                 logger.warning("Failed to update MongoDB with MinIO path: %s", exc)

#     qdrant_chunks = _index_in_vectorstore(
#         text=extraction.text,
#         title=doc_type,
#         doc_type=doc_type,
#         case_id=case_id,
#         source_file=file_path,
#         mongo_id=str(mongo_id) if mongo_id else "",
#         file_id=file_id,
#     )

#     logger.info(
#         "Processed '%s': type='%s' confidence=%d mongo_id=%s chunks=%d words=%d",
#         file_path, doc_type, confidence, mongo_id, qdrant_chunks,
#         len(extraction.word_confidences),
#     )

#     return {
#         "text": extraction.text,
#         "file_type": file_type,
#         "classification": classification,
#         "metadata": {
#             "mongo_id": str(mongo_id) if mongo_id else None,
#             "minio_object": minio_object,
#             "qdrant_chunks": qdrant_chunks,
#             "case_id": case_id,
#             "source_file": file_path,
#             "file_id": file_id,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Reindex (called from document_service on OCR correction)
# # ---------------------------------------------------------------------------

# def reindex_document(
#     mongo_id: str,
#     new_text: str,
#     doc_meta: Dict[str, Any],
# ) -> int:
#     """Delete existing Qdrant chunks for mongo_id and re-index new_text.

#     ``doc_meta`` must contain: title, doc_type, case_id, source_file.
#     Optional: file_id.
#     Returns the number of new chunks indexed.
#     """
#     _delete_qdrant_chunks_by_mongo_id(mongo_id)
#     return _index_in_vectorstore(
#         text=new_text,
#         title=doc_meta.get("title", ""),
#         doc_type=doc_meta.get("doc_type", ""),
#         case_id=doc_meta.get("case_id", ""),
#         source_file=doc_meta.get("source_file", ""),
#         mongo_id=mongo_id,
#         file_id=doc_meta.get("file_id"),
#     )


"""
DocumentProcessor.pipeline
---------------------------
Unified document processing: ingest → OCR/extract → classify → store.

Public API
----------
    process_document(file_path, case_id, file_id)       -> dict
    process_document_group(file_paths, case_id, file_ids) -> dict
    reindex_document(mongo_id, new_text, doc_meta)      -> int

OCR changes (GCV migration)
----------------------------
``_extract_text_via_ocr`` now returns an ``_OCRExtraction`` dataclass
instead of a bare string.  It carries:

    text            — canonical text (LLM-refined if refinement enabled,
                      otherwise normalized OCR text).  This is what gets
                      classified and indexed.
    raw_ocr_text    — verbatim GCV output, stored in MongoDB for audit.
                      NOT exposed in the API.
    word_confidences — list[dict] ready for MongoDB storage and API responses.

Word confidence dicts follow the schema::

    {"word": str, "confidence": float, "band": "high"|"mid"|"low",
     "page_number": int}
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from config.supervisor import (
    EMBEDDING_MODEL,
    MONGO_COLLECTION,
    MONGO_DB,
    MONGO_URI,
    QDRANT_COLLECTION_CASE,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
)
from DocumentProcessor.classifier import classify_document
from DocumentProcessor.OCR.ocr_pipeline import run_ocr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File type constants
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {".txt", ".text", ".csv", ".json", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

_MAGIC_BYTES = {
    b"%PDF": "pdf",
    b"\x89PNG": "image",
    b"\xff\xd8\xff": "image",
    b"GIF8": "image",
    b"BM": "image",
    b"RIFF": "image",
    b"\x49\x49\x2a\x00": "image",
    b"\x4d\x4d\x00\x2a": "image",
}

PAGE_SEP = "\n\n--- PAGE BREAK ---\n\n"

# ---------------------------------------------------------------------------
# OCR extraction result
# ---------------------------------------------------------------------------


@dataclass
class _OCRExtraction:
    """Structured result from the OCR + LLM refinement pipeline."""

    text: str                                  # canonical (refined or normalized)
    raw_ocr_text: str = ""                     # verbatim GCV output (audit only)
    word_confidences: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy singletons — double-checked locking
# ---------------------------------------------------------------------------

_mongo_client: Optional[MongoClient] = None
_mongo_lock = threading.Lock()

_vectorstore = None
_vectorstore_lock = threading.Lock()

_minio_config: Dict[str, Any] = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "bucket": os.getenv("MINIO_BUCKET", "hakim-files"),
    "secure": False,
}


def _get_mongo_collection():
    global _mongo_client
    if _mongo_client is None:
        with _mongo_lock:
            if _mongo_client is None:
                _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[MONGO_DB][MONGO_COLLECTION]


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    grpc_port=QDRANT_GRPC_PORT,
                    prefer_grpc=QDRANT_PREFER_GRPC,
                )

                existing = [c.name for c in client.get_collections().collections]
                if QDRANT_COLLECTION_CASE not in existing:
                    try:
                        client.create_collection(
                            collection_name=QDRANT_COLLECTION_CASE,
                            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                        )
                        logger.info("Created Qdrant collection '%s'", QDRANT_COLLECTION_CASE)
                    except Exception as exc:
                        logger.info(
                            "Collection '%s' already exists (concurrent create): %s",
                            QDRANT_COLLECTION_CASE, exc,
                        )

                _vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name=QDRANT_COLLECTION_CASE,
                    embedding=embeddings,
                )
    return _vectorstore


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def detect_file_type(file_path: str) -> str:
    """Return ``'text'``, ``'pdf'``, ``'image'``, or ``'unknown'``."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        for magic, ftype in _MAGIC_BYTES.items():
            if header.startswith(magic):
                return ftype
    except OSError:
        pass

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_file(file_path: str) -> str:
    for encoding in ("utf-8", "cp1256", "windows-1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        logger.warning("File '%s' decoded with replacement chars", file_path)
        return f.read()


def _extract_text_from_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf required for PDF extraction. pip install pypdf")
        return ""

    try:
        reader = PdfReader(file_path)
        pages_text = [
            page.extract_text().strip()
            for page in reader.pages
            if page.extract_text()
        ]
        result = "\n\n".join(pages_text)
        if not result:
            logger.warning(
                "PDF '%s' extracted to empty text — may be a scanned/image PDF",
                file_path,
            )
        return result
    except Exception as exc:
        logger.exception("Failed to extract text from PDF '%s': %s", file_path, exc)
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def _extract_text_via_ocr(
    file_path: str,
    doc_id: Optional[str] = None,
) -> _OCRExtraction:
    """Run the OCR + LLM pipeline and return an ``_OCRExtraction``.

    The returned ``text`` field is the LLM-refined text (or normalized OCR
    text when refinement is disabled / fails).  ``raw_ocr_text`` is the
    verbatim GCV output stored for audit purposes.  ``word_confidences``
    is a list of serialisable dicts suitable for MongoDB storage.
    """
    try:
        result = run_ocr(file_path=file_path, doc_id=doc_id)

        all_word_confidences: List[Dict] = []
        canonical_pages: List[str] = []
        raw_pages: List[str] = []

        for page in result.pages:
            if not page.raw_text:
                continue

            # Canonical text: prefer refined; fall back to normalized
            canonical_pages.append(page.canonical_text)
            raw_pages.append(page.raw_text)

            if page.word_confidences:
                for wc in page.word_confidences:
                    all_word_confidences.append(wc.model_dump())

        return _OCRExtraction(
            text="\n\n".join(canonical_pages),
            raw_ocr_text="\n\n".join(raw_pages),
            word_confidences=all_word_confidences,
        )

    except Exception as exc:
        logger.exception("OCR pipeline failed for '%s': %s", file_path, exc)
        return _OCRExtraction(text="", raw_ocr_text="", word_confidences=[])


def _extract_text(
    file_path: str,
    file_type: str,
    case_id: str,
) -> _OCRExtraction:
    """Extract text from any supported file type.

    Always returns ``_OCRExtraction`` for a uniform caller interface.
    Non-OCR paths (text, PDF) set ``raw_ocr_text`` and ``word_confidences``
    to empty values.
    """
    if file_type == "text":
        return _OCRExtraction(text=_extract_text_from_file(file_path))
    if file_type == "pdf":
        return _OCRExtraction(text=_extract_text_from_pdf(file_path))
    if file_type == "image":
        return _extract_text_via_ocr(file_path, doc_id=case_id)

    # Unknown — try plain text as a best-effort fallback
    try:
        return _OCRExtraction(text=_extract_text_from_file(file_path))
    except Exception:
        return _OCRExtraction(text="")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _store_in_mongo(
    title: str,
    doc_type: str,
    case_id: str,
    source_file: str,
    text: str,
    confidence: int,
    explanation: str,
    file_type: str,
    raw_ocr_text: str = "",
    word_confidences: Optional[List[Dict]] = None,
    file_id: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    source_files: Optional[List[str]] = None,
) -> Optional[Any]:
    source_file = source_file.replace("\\", "/")
    doc_record: Dict[str, Any] = {
        "title": title,
        "doc_type": doc_type,
        "case_id": case_id,
        "source_file": source_file,
        "text": text,                               # canonical (refined) text
        "raw_ocr_text": raw_ocr_text,              # verbatim GCV output
        "word_confidences": word_confidences or [], # per-word confidence for UI
        "classification_confidence": confidence,
        "classification_explanation": explanation,
        "file_type": file_type,
        "file_id": file_id,
        "storage_backend": "local",
        "minio_object": None,
        "created_at": datetime.now(timezone.utc),
        "corrected": False,
    }
    if file_ids is not None:
        doc_record["file_ids"] = file_ids
    if source_files is not None:
        doc_record["source_files"] = [sf.replace("\\", "/") for sf in source_files]

    try:
        col = _get_mongo_collection()

        # Deduplication only for the legacy single-file path
        if file_ids is None:
            existing = col.find_one(
                {"source_file": source_file, "case_id": case_id},
                {"_id": 1},
            )
            if existing:
                logger.info(
                    "Skipping duplicate ingest: source_file='%s' case_id='%s'",
                    source_file, case_id,
                )
                return existing["_id"]

        result = col.insert_one(doc_record)
        logger.info("Stored in MongoDB: title='%s' id=%s", title, result.inserted_id)
        return result.inserted_id

    except Exception as exc:
        logger.exception("MongoDB insert failed for '%s': %s", title, exc)
        return None


def _upload_to_minio(file_path: str, mongo_id: str, case_id: str) -> Optional[str]:
    try:
        from minio import Minio

        cfg = _minio_config
        client = Minio(
            endpoint=cfg["endpoint"],
            access_key=cfg["access_key"],
            secret_key=cfg["secret_key"],
            secure=cfg["secure"],
        )

        if not client.bucket_exists(cfg["bucket"]):
            try:
                client.make_bucket(cfg["bucket"])
            except Exception:
                pass

        filename = os.path.basename(file_path)
        safe_case_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", case_id) if case_id else "no-case"
        object_name = f"{safe_case_id}/{mongo_id}/{filename}"

        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            client.put_object(
                bucket_name=cfg["bucket"],
                object_name=object_name,
                data=f,
                length=file_size,
            )

        logger.info("Uploaded to MinIO: %s/%s", cfg["bucket"], object_name)
        return object_name

    except Exception as exc:
        logger.warning("MinIO upload failed (non-fatal): %s", exc)
        return None


def _index_in_vectorstore(
    text: str,
    title: str,
    doc_type: str,
    case_id: str,
    source_file: str,
    mongo_id: str,
    file_id: Optional[str] = None,
) -> int:
    """Chunk and index text. Returns the number of chunks indexed."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(text)
    if not chunks:
        logger.warning("No chunks produced from text for '%s'", source_file)
        return 0

    metadatas = [
        {
            "title": title,
            "type": doc_type,
            "case_id": case_id,
            "source_file": source_file,
            "mongo_id": mongo_id,
            "file_id": file_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    try:
        _get_vectorstore().add_texts(texts=chunks, metadatas=metadatas)
        logger.info(
            "Indexed %d chunk(s) for '%s' (case=%s)", len(chunks), title, case_id
        )
        return len(chunks)
    except Exception as exc:
        logger.exception(
            "Vector store indexing failed for '%s': %s", source_file, exc
        )
        return 0


def _delete_qdrant_chunks_by_mongo_id(mongo_id: str) -> None:
    """Delete all Qdrant points where metadata.mongo_id == mongo_id."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    vs = _get_vectorstore()
    client = vs.client
    filt = Filter(
        must=[FieldCondition(key="metadata.mongo_id", match=MatchValue(value=mongo_id))]
    )
    client.delete(collection_name=vs.collection_name, points_selector=filt)
    logger.info("Deleted Qdrant chunks for mongo_id=%s", mongo_id)


# ---------------------------------------------------------------------------
# Public API — multi-file group (primary ingestion path)
# ---------------------------------------------------------------------------

def process_document_group(
    file_paths: List[str],
    case_id: str,
    file_ids: List[str],
) -> Dict[str, Any]:
    """Process a group of files as one multi-page document.

    Files are processed individually in order; their canonical texts are
    joined with ``PAGE_SEP``.  Word confidences from all pages are merged
    into a flat list (each carries a ``page_number`` relative to its OCR run).
    Classification and Qdrant indexing happen once on the merged text.
    """
    per_file_extractions: List[_OCRExtraction] = []
    file_types: List[str] = []

    for path in file_paths:
        ft = detect_file_type(path)
        file_types.append(ft)
        extraction = _extract_text(path, ft, case_id)
        per_file_extractions.append(extraction)

    merged_text = PAGE_SEP.join(
        e.text for e in per_file_extractions if e.text
    ).strip()

    merged_raw_ocr = PAGE_SEP.join(
        e.raw_ocr_text for e in per_file_extractions if e.raw_ocr_text
    ).strip()

    merged_word_confidences: List[Dict] = [
        wc
        for extraction in per_file_extractions
        for wc in extraction.word_confidences
    ]

    unique_types = set(file_types)
    composite_type = unique_types.pop() if len(unique_types) == 1 else "mixed"

    source_files = [os.path.basename(p) for p in file_paths]
    primary_source = source_files[0] if source_files else ""
    primary_file_id = file_ids[0] if file_ids else None

    _empty_meta = {
        "mongo_id": None,
        "minio_object": None,
        "qdrant_chunks": 0,
        "case_id": case_id,
        "source_file": primary_source,
        "source_files": source_files,
        "file_id": primary_file_id,
        "file_ids": file_ids,
    }

    if not merged_text:
        logger.warning("No text extracted from group %s", file_ids)
        return {
            "text": "",
            "file_type": composite_type,
            "classification": {
                "final_type": "مستند غير معروف",
                "confidence": 0,
                "explanation": "No text could be extracted",
            },
            "metadata": _empty_meta,
        }

    classification = classify_document(merged_text)
    doc_type = classification.get("final_type") or "مستند غير معروف"
    confidence = classification.get("confidence", 0)
    explanation = classification.get("explanation", "")

    mongo_id = _store_in_mongo(
        title=doc_type,
        doc_type=doc_type,
        case_id=case_id,
        source_file=primary_source,
        text=merged_text,
        confidence=confidence,
        explanation=explanation,
        file_type=composite_type,
        raw_ocr_text=merged_raw_ocr,
        word_confidences=merged_word_confidences,
        file_id=primary_file_id,
        file_ids=file_ids,
        source_files=source_files,
    )

    qdrant_chunks = _index_in_vectorstore(
        text=merged_text,
        title=doc_type,
        doc_type=doc_type,
        case_id=case_id,
        source_file=primary_source,
        mongo_id=str(mongo_id) if mongo_id else "",
        file_id=primary_file_id,
    )

    logger.info(
        "Group %s processed: type='%s' confidence=%d mongo_id=%s chunks=%d words=%d",
        file_ids, doc_type, confidence, mongo_id, qdrant_chunks,
        len(merged_word_confidences),
    )

    return {
        "text": merged_text,
        "file_type": composite_type,
        "classification": classification,
        "metadata": {
            "mongo_id": str(mongo_id) if mongo_id else None,
            "minio_object": None,
            "qdrant_chunks": qdrant_chunks,
            "case_id": case_id,
            "source_file": primary_source,
            "source_files": source_files,
            "file_id": primary_file_id,
            "file_ids": file_ids,
        },
    }


# ---------------------------------------------------------------------------
# Public API — single document (legacy path)
# ---------------------------------------------------------------------------

def process_document(
    file_path: str,
    case_id: str = "",
    file_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single document end-to-end.

    Returns
    -------
    dict with keys:
        text           -- canonical extracted text (refined if OCR)
        file_type      -- 'text' | 'pdf' | 'image' | 'unknown'
        classification -- {final_type, confidence, explanation}
        metadata       -- {mongo_id, minio_object, qdrant_chunks, …}
    """
    file_type = detect_file_type(file_path)
    extraction = _extract_text(file_path, file_type, case_id)

    _empty_meta = {
        "mongo_id": None,
        "minio_object": None,
        "qdrant_chunks": 0,
        "case_id": case_id,
        "source_file": file_path,
        "file_id": file_id,
    }

    if not extraction.text or not extraction.text.strip():
        logger.warning("No text extracted from '%s'", file_path)
        return {
            "text": "",
            "file_type": file_type,
            "classification": {
                "final_type": "مستند غير معروف",
                "confidence": 0,
                "explanation": "No text could be extracted",
            },
            "metadata": _empty_meta,
        }

    classification = classify_document(extraction.text)
    doc_type = classification.get("final_type") or "مستند غير معروف"
    confidence = classification.get("confidence", 0)
    explanation = classification.get("explanation", "")

    mongo_id = _store_in_mongo(
        title=doc_type,
        doc_type=doc_type,
        case_id=case_id,
        source_file=file_path,
        text=extraction.text,
        confidence=confidence,
        explanation=explanation,
        file_type=file_type,
        raw_ocr_text=extraction.raw_ocr_text,
        word_confidences=extraction.word_confidences,
        file_id=file_id,
    )

    minio_object = None
    if mongo_id and file_type != "unknown":
        minio_object = _upload_to_minio(
            file_path=file_path,
            mongo_id=str(mongo_id),
            case_id=case_id,
        )
        if minio_object:
            try:
                _get_mongo_collection().update_one(
                    {"_id": mongo_id},
                    {"$set": {"minio_object": minio_object, "storage_backend": "minio"}},
                )
            except Exception as exc:
                logger.warning("Failed to update MongoDB with MinIO path: %s", exc)

    qdrant_chunks = _index_in_vectorstore(
        text=extraction.text,
        title=doc_type,
        doc_type=doc_type,
        case_id=case_id,
        source_file=file_path,
        mongo_id=str(mongo_id) if mongo_id else "",
        file_id=file_id,
    )

    logger.info(
        "Processed '%s': type='%s' confidence=%d mongo_id=%s chunks=%d words=%d",
        file_path, doc_type, confidence, mongo_id, qdrant_chunks,
        len(extraction.word_confidences),
    )

    return {
        "text": extraction.text,
        "file_type": file_type,
        "classification": classification,
        "metadata": {
            "mongo_id": str(mongo_id) if mongo_id else None,
            "minio_object": minio_object,
            "qdrant_chunks": qdrant_chunks,
            "case_id": case_id,
            "source_file": file_path,
            "file_id": file_id,
        },
    }


# ---------------------------------------------------------------------------
# Reindex (called from document_service on OCR correction)
# ---------------------------------------------------------------------------

def reindex_document(
    mongo_id: str,
    new_text: str,
    doc_meta: Dict[str, Any],
) -> int:
    """Delete existing Qdrant chunks for mongo_id and re-index new_text.

    ``doc_meta`` must contain: title, doc_type, case_id, source_file.
    Optional: file_id.
    Returns the number of new chunks indexed.
    """
    _delete_qdrant_chunks_by_mongo_id(mongo_id)
    return _index_in_vectorstore(
        text=new_text,
        title=doc_meta.get("title", ""),
        doc_type=doc_meta.get("doc_type", ""),
        case_id=doc_meta.get("case_id", ""),
        source_file=doc_meta.get("source_file", ""),
        mongo_id=mongo_id,
        file_id=doc_meta.get("file_id"),
    )