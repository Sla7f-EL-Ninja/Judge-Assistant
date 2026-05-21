"""
DocumentProcessor.pipeline
---------------------------
Unified document processing: detect type → extract text → classify → store.

Public API:
    process_document(file_path, case_id="", file_id=None) -> dict
    reindex_document(mongo_id, new_text, doc_meta) -> int
""" 

import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from config.supervisor import (
    EMBEDDING_MODEL,
    MONGO_COLLECTION,
    MONGO_DB,
    MONGO_URI,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_GRPC_PORT,
    QDRANT_PREFER_GRPC,
    QDRANT_COLLECTION_CASE,
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

# ---------------------------------------------------------------------------
# Lazy singletons — double-checked locking for thread safety
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
                        logger.info("Collection '%s' already exists (concurrent create): %s", QDRANT_COLLECTION_CASE, exc)

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
# Text extraction helpers
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
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        result = "\n\n".join(pages_text)
        if not result:
            logger.warning("PDF '%s' extracted to empty text (scanned/image PDF?)", file_path)
        return result
    except Exception as exc:
        logger.exception("Failed to extract text from PDF '%s': %s", file_path, exc)
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def _extract_text_via_ocr(file_path: str, doc_id: Optional[str] = None) -> str:
    try:
        result = run_ocr(file_path=file_path, doc_id=doc_id)
        return "\n\n".join(p.raw_text for p in result.pages if p.raw_text)
    except Exception as exc:
        logger.exception("OCR failed for '%s': %s", file_path, exc)
        return ""


def _extract_text(file_path: str, file_type: str, case_id: str) -> str:
    if file_type == "text":
        return _extract_text_from_file(file_path)
    elif file_type == "pdf":
        return _extract_text_from_pdf(file_path)
    elif file_type == "image":
        return _extract_text_via_ocr(file_path, doc_id=case_id)
    else:
        try:
            return _extract_text_from_file(file_path)
        except Exception:
            return ""


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
    file_id: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    source_files: Optional[List[str]] = None,
) -> Optional[Any]:
    source_file = source_file.replace("\\", "/")
    doc_record = {
        "title": title,
        "doc_type": doc_type,
        "case_id": case_id,
        "source_file": source_file,
        "text": text,
        "original_text": text,  # preserved forever; never overwritten by corrections
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
        # Dedupe only for single-file ingestion (legacy path)
        if file_ids is None:
            existing = col.find_one(
                {"source_file": source_file, "case_id": case_id},
                {"_id": 1},
            )
            if existing:
                logger.info(
                    "Skipping duplicate ingest for source_file='%s' case_id='%s'",
                    source_file, case_id,
                )
                return existing["_id"]

        result = col.insert_one(doc_record)
        logger.info("Stored in MongoDB: title='%s', id=%s", title, result.inserted_id)
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
    """Chunk and index text. Returns number of chunks indexed."""
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
        logger.exception("Vector store indexing failed for '%s': %s", source_file, exc)
        return 0


def _delete_qdrant_chunks_by_mongo_id(mongo_id: str) -> None:
    """Delete all Qdrant points where metadata.mongo_id == mongo_id."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    vs = _get_vectorstore()
    client = vs.client
    collection = vs.collection_name
    filt = Filter(
        must=[FieldCondition(key="metadata.mongo_id", match=MatchValue(value=mongo_id))]
    )
    client.delete(collection_name=collection, points_selector=filt)
    logger.info("Deleted Qdrant chunks for mongo_id=%s", mongo_id)


def process_document_group(
    file_paths: List[str],
    case_id: str,
    file_ids: List[str],
) -> Dict[str, Any]:
    """Process a group of files as a single multi-page document.

    Each file is OCR'd / text-extracted individually; results are concatenated in
    order with a page-break separator. Classification and indexing happen once on
    the merged text.

    Returns same shape as process_document: {text, file_type, classification, metadata}.
    """
    PAGE_SEP = "\n\n--- PAGE BREAK ---\n\n"

    per_file_texts: List[str] = []
    file_types: List[str] = []

    for path in file_paths:
        ft = detect_file_type(path)
        file_types.append(ft)
        text = _extract_text(path, ft, case_id)
        per_file_texts.append(text or "")

    merged_text = PAGE_SEP.join(per_file_texts).strip()

    # Determine composite file_type label
    unique_types = set(file_types)
    if len(unique_types) == 1:
        composite_type = unique_types.pop()
    else:
        composite_type = "mixed"

    source_files = [os.path.basename(p) for p in file_paths]
    primary_source = source_files[0] if source_files else ""
    primary_file_id = file_ids[0] if file_ids else None

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
            "metadata": {
                "mongo_id": None,
                "minio_object": None,
                "qdrant_chunks": 0,
                "case_id": case_id,
                "source_file": primary_source,
                "source_files": source_files,
                "file_id": primary_file_id,
                "file_ids": file_ids,
            },
        }

    classification = classify_document(merged_text)
    doc_type = classification.get("final_type") or "مستند غير معروف"
    confidence = classification.get("confidence", 0)
    explanation = classification.get("explanation", "")
    title = doc_type

    mongo_id = _store_in_mongo(
        title=title,
        doc_type=doc_type,
        case_id=case_id,
        source_file=primary_source,
        text=merged_text,
        confidence=confidence,
        explanation=explanation,
        file_type=composite_type,
        file_id=primary_file_id,
        file_ids=file_ids,
        source_files=source_files,
    )

    # Files are already in MinIO via the upload endpoint — skip _upload_to_minio.

    qdrant_chunks = _index_in_vectorstore(
        text=merged_text,
        title=title,
        doc_type=doc_type,
        case_id=case_id,
        source_file=primary_source,
        mongo_id=str(mongo_id) if mongo_id else "",
        file_id=primary_file_id,
    )

    logger.info(
        "Processed group %s: type='%s', confidence=%d, mongo_id=%s, chunks=%d",
        file_ids, doc_type, confidence, mongo_id, qdrant_chunks,
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


def reindex_document(mongo_id: str, new_text: str, doc_meta: Dict[str, Any]) -> int:
    """Delete existing Qdrant chunks for mongo_id and re-index new_text.

    doc_meta must contain: title, doc_type, case_id, source_file. Optional: file_id.
    Returns number of new chunks indexed.
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


# ---------------------------------------------------------------------------
# Public API
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
        text          -- extracted text content
        file_type     -- 'text' | 'pdf' | 'image' | 'unknown'
        classification -- {final_type, confidence, explanation}
        metadata      -- {mongo_id, minio_object, qdrant_chunks, case_id, source_file, file_id}
    """
    file_type = detect_file_type(file_path)

    text = _extract_text(file_path, file_type, case_id)

    if not text or not text.strip():
        logger.warning("No text extracted from '%s'", file_path)
        return {
            "text": "",
            "file_type": file_type,
            "classification": {
                "final_type": "مستند غير معروف",
                "confidence": 0,
                "explanation": "No text could be extracted",
            },
            "metadata": {
                "mongo_id": None,
                "minio_object": None,
                "qdrant_chunks": 0,
                "case_id": case_id,
                "source_file": file_path,
                "file_id": file_id,
            },
        }

    classification = classify_document(text)
    doc_type = classification.get("final_type") or "مستند غير معروف"
    confidence = classification.get("confidence", 0)
    explanation = classification.get("explanation", "")
    title = doc_type

    mongo_id = _store_in_mongo(
        title=title,
        doc_type=doc_type,
        case_id=case_id,
        source_file=file_path,
        text=text,
        confidence=confidence,
        explanation=explanation,
        file_type=file_type,
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
        text=text,
        title=title,
        doc_type=doc_type,
        case_id=case_id,
        source_file=file_path,
        mongo_id=str(mongo_id) if mongo_id else "",
        file_id=file_id,
    )

    logger.info(
        "Processed '%s': type='%s', confidence=%d, mongo_id=%s, chunks=%d",
        file_path, doc_type, confidence, mongo_id, qdrant_chunks,
    )

    return {
        "text": text,
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