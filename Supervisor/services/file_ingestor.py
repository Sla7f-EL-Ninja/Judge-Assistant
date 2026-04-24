"""
file_ingestor.py

Unified file ingestion service that handles text, PDF, and image files.

Workflow per file:
1. Detect file type (text / PDF / image).
2. Extract text:
   - Text files (.txt, .text, .csv, .json, .md) -> read directly.
   - PDF files (.pdf) -> extract text with PyPDF2.
   - Image files (.png, .jpg, .jpeg, .tiff, .bmp, .webp) -> run OCR.
3. Classify the document using the document classifier.
4. Store the document record in MongoDB.
4.5 Upload the raw file to MinIO (non-fatal fallback if MinIO is down).
5. Index the document text in the Qdrant vector store so the Case Doc RAG
   can retrieve it dynamically.

This service can be called:
- **Before** a case run, to pre-load documents into the system.
- **During** a run, from the classify_and_store_document node.
"""

import io
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File type constants
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {".txt", ".text", ".csv", ".json", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies
# ---------------------------------------------------------------------------

def _get_classifier():
    """Lazy-import the document classifier."""
    classifier_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "RAG", "Case Doc RAG",
    )
    classifier_dir = os.path.normpath(classifier_dir)
    if classifier_dir not in sys.path:
        sys.path.insert(0, classifier_dir)

    from document_classifier import classify_document
    return classify_document


def _get_ocr_processor():
    """Lazy-import the OCR pipeline's process_document function."""
    ocr_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "OCR",
    )
    ocr_dir = os.path.normpath(ocr_dir)
    if ocr_dir not in sys.path:
        sys.path.insert(0, ocr_dir)

    from ocr_pipeline import process_document
    return process_document


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

_MAGIC_BYTES = {
    b"%PDF": "pdf",
    b"\x89PNG": "image",
    b"\xff\xd8\xff": "image",   # JPEG
    b"GIF8": "image",
    b"BM": "image",             # BMP
    b"RIFF": "image",           # WebP (RIFF....WEBP)
    b"\x49\x49\x2a\x00": "image",  # TIFF LE
    b"\x4d\x4d\x00\x2a": "image",  # TIFF BE
}


def detect_file_type(file_path: str) -> str:
    """Return ``'text'``, ``'pdf'``, ``'image'``, or ``'unknown'``.

    Checks magic bytes first, then falls back to extension (B13).
    """
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

def extract_text_from_file(file_path: str) -> str:
    """Read a plain text file; tries UTF-8 then CP1256 for Arabic docs."""
    for encoding in ("utf-8", "cp1256", "windows-1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Last resort: replace errors
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        logger.warning("File '%s' decoded with replacement chars — check encoding", file_path)
        return f.read()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2.

    Falls back to an empty string if extraction fails.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error(
            "PyPDF2 is required for PDF text extraction. "
            "Install it with: pip install PyPDF2"
        )
        return ""

    try:
        reader = PdfReader(file_path)
        pages_text: List[str] = []
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


def extract_text_via_ocr(file_path: str, doc_id: Optional[str] = None) -> str:
    """Run the OCR pipeline on an image file and return extracted text."""
    try:
        process_document = _get_ocr_processor()
        result = process_document(file_path=file_path, doc_id=doc_id)
        return result.raw_text
    except Exception as exc:
        logger.exception("OCR failed for '%s': %s", file_path, exc)
        return ""


# ---------------------------------------------------------------------------
# FileIngestor - main service class
# ---------------------------------------------------------------------------

class FileIngestor:
    """Handles end-to-end ingestion of files into MongoDB, MinIO, and the vector store.

    Parameters
    ----------
    mongo_uri : str
        MongoDB connection URI.
    mongo_db : str
        Database name (default ``"Rag"``).
    mongo_collection : str
        Collection name (default ``"Document Storage"``).
    embedding_model : str
        HuggingFace embedding model for the vector store.
    chroma_collection : str
        (Deprecated) Alias for ``qdrant_collection``.  Kept for
        backward compatibility; prefer ``qdrant_collection``.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "Rag",
        mongo_collection: str = "Document Storage",
        embedding_model: str = "BAAI/bge-m3",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_grpc_port: int = 6334,
        qdrant_prefer_grpc: bool = True,
        qdrant_collection: Optional[str] = None,
        minio_endpoint: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
        minio_bucket: Optional[str] = None,
        minio_secure: bool = False,
    ):
        self._mongo_uri = mongo_uri or os.getenv(
            "MONGO_URI", "mongodb://localhost:27017/"
        )
        self._mongo_db_name = mongo_db
        self._mongo_col_name = mongo_collection
        self._embedding_model_name = embedding_model

        # Qdrant config (replaces ChromaDB)
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._qdrant_grpc_port = qdrant_grpc_port
        self._qdrant_prefer_grpc = qdrant_prefer_grpc
        self._qdrant_collection_name = qdrant_collection 

        # MinIO config
        self._minio_endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self._minio_access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self._minio_secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self._minio_bucket = minio_bucket or os.getenv("MINIO_BUCKET", "hakim-files")
        self._minio_secure = minio_secure

        # Lazily initialised (locks prevent first-call races under concurrency)
        self._mongo_client: Optional[MongoClient] = None
        self._vectorstore = None
        self._classifier = None
        self._mongo_lock = threading.Lock()
        self._vectorstore_lock = threading.Lock()
        self._classifier_lock = threading.Lock()

    # -- Lazy accessors ---------------------------------------------------

    @property
    def mongo_collection(self):
        """Return the MongoDB collection, connecting if needed."""
        if self._mongo_client is None:
            with self._mongo_lock:
                if self._mongo_client is None:
                    self._mongo_client = MongoClient(self._mongo_uri)
        db = self._mongo_client[self._mongo_db_name]
        return db[self._mongo_col_name]

    @property
    def vectorstore(self):
        """Return the Qdrant vector store, creating collection if needed."""
        if self._vectorstore is None:
            with self._vectorstore_lock:
                if self._vectorstore is None:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    from langchain_qdrant import QdrantVectorStore
                    from qdrant_client import QdrantClient
                    from qdrant_client.models import Distance, VectorParams

                    embeddings = HuggingFaceEmbeddings(
                        model_name=self._embedding_model_name,
                    )

                    client = QdrantClient(
                        host=self._qdrant_host,
                        port=self._qdrant_port,
                        grpc_port=self._qdrant_grpc_port,
                        prefer_grpc=self._qdrant_prefer_grpc,
                    )

                    # Create collection if it doesn't exist; tolerate race with
                    # another worker that created it between our check and create.
                    existing = [c.name for c in client.get_collections().collections]
                    if self._qdrant_collection_name not in existing:
                        try:
                            client.create_collection(
                                collection_name=self._qdrant_collection_name,
                                vectors_config=VectorParams(
                                    size=1024,  # BAAI/bge-m3 output dimension
                                    distance=Distance.COSINE,
                                ),
                            )
                            logger.info("Created Qdrant collection '%s'", self._qdrant_collection_name)
                        except Exception as exc:
                            # Another worker created it concurrently — safe to continue
                            logger.info("Collection '%s' already exists (concurrent create): %s", self._qdrant_collection_name, exc)

                    self._vectorstore = QdrantVectorStore(
                        client=client,
                        collection_name=self._qdrant_collection_name,
                        embedding=embeddings,
                    )
        return self._vectorstore

    @property
    def classifier(self):
        """Return the document classifier function."""
        if self._classifier is None:
            with self._classifier_lock:
                if self._classifier is None:
                    self._classifier = _get_classifier()
        return self._classifier

    # -- Core ingestion ---------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        case_id: str = "",
        pre_extracted_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a single file end-to-end.

        Parameters
        ----------
        file_path : str
            Path to the file on disk.
        case_id : str
            The case this document belongs to.
        pre_extracted_text : str, optional
            If provided, skip text extraction and use this text directly
            (used when OCR has already been run upstream).

        Returns
        -------
        dict
            Ingestion result with keys: file, title, doc_type, confidence,
            explanation, mongo_id, minio_object, file_type.
        """
        file_type = detect_file_type(file_path)

        # 1. Extract text
        if pre_extracted_text is not None:
            text = pre_extracted_text
        else:
            text = self._extract_text(file_path, file_type, case_id)

        if not text or not text.strip():
            logger.warning("No text extracted from '%s'", file_path)
            return {
                "file": file_path,
                "title": "",
                "doc_type": "unknown",
                "confidence": 0,
                "explanation": "No text could be extracted",
                "mongo_id": None,
                "minio_object": None,
                "file_type": file_type,
            }

        # 2. Classify
        classification = self.classifier(text)
        doc_type = classification.get("final_type", "مستند غير معروف")
        confidence = classification.get("confidence", 0)
        explanation = classification.get("explanation", "")
        title = doc_type

        # 3. Store in MongoDB
        mongo_id = self._store_in_mongo(
            title=title,
            doc_type=doc_type,
            case_id=case_id,
            source_file=file_path,
            text=text,
            confidence=confidence,
            explanation=explanation,
            file_type=file_type,
        )

        # 3.5 Upload raw file to MinIO (non-fatal)
        minio_object = None
        if mongo_id and file_type != "unknown":
            minio_object = self._upload_to_minio(
                file_path=file_path,
                mongo_id=str(mongo_id),
                case_id=case_id,
            )
            # Patch the MongoDB record with the MinIO object name
            if minio_object:
                try:
                    self.mongo_collection.update_one(
                        {"_id": mongo_id},
                        {"$set": {
                            "minio_object": minio_object,
                            "storage_backend": "minio",
                        }},
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to update MongoDB with MinIO path: %s", exc
                    )

        # 4. Index in vector store
        self._index_in_vectorstore(
            text=text,
            title=title,
            doc_type=doc_type,
            case_id=case_id,
            source_file=file_path,
            mongo_id=str(mongo_id) if mongo_id else "",
        )

        logger.info(
            "Ingested '%s': type='%s', confidence=%d, mongo_id=%s",
            file_path, doc_type, confidence, mongo_id,
        )

        return {
            "file": file_path,
            "title": title,
            "doc_type": doc_type,
            "confidence": confidence,
            "explanation": explanation,
            "mongo_id": str(mongo_id) if mongo_id else None,
            "minio_object": minio_object,
            "file_type": file_type,
        }

    def ingest_files(
        self,
        file_paths: List[str],
        case_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Ingest multiple files. Convenience wrapper around ``ingest_file``.

        Parameters
        ----------
        file_paths : list of str
            Paths to files on disk.
        case_id : str
            The case these documents belong to.

        Returns
        -------
        list of dict
            One result dict per file.
        """
        results: List[Dict[str, Any]] = [None] * len(file_paths)  # preserve order

        def _ingest_one(idx: int, fp: str) -> tuple:
            try:
                return idx, self.ingest_file(fp, case_id=case_id)
            except Exception as exc:
                logger.exception("Failed to ingest '%s': %s", fp, exc)
                return idx, {
                    "file": fp,
                    "title": "",
                    "doc_type": "unknown",
                    "confidence": 0,
                    "explanation": f"Ingestion failed: {exc}",
                    "mongo_id": None,
                    "minio_object": None,
                    "file_type": detect_file_type(fp),
                }

        max_workers = min(4, len(file_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_ingest_one, i, fp): i for i, fp in enumerate(file_paths)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def ingest_ocr_results(
        self,
        raw_texts: List[str],
        uploaded_files: List[str],
        case_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Ingest pre-extracted OCR texts (already processed by the OCR adapter).

        Parameters
        ----------
        raw_texts : list of str
            Text strings extracted by the OCR pipeline.
        uploaded_files : list of str
            Corresponding file paths for reference.
        case_id : str
            The case these documents belong to.

        Returns
        -------
        list of dict
            One result dict per text.
        """
        results = []
        for i, text in enumerate(raw_texts):
            file_ref = uploaded_files[i] if i < len(uploaded_files) else f"ocr_doc_{i}"
            if not text or not text.strip():
                logger.warning("Skipping empty OCR text at index %d", i)
                continue
            try:
                result = self.ingest_file(
                    file_path=file_ref,
                    case_id=case_id,
                    pre_extracted_text=text,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("Failed to ingest OCR text %d: %s", i, exc)
                results.append({
                    "file": file_ref,
                    "title": "",
                    "doc_type": "unknown",
                    "confidence": 0,
                    "explanation": f"Ingestion failed: {exc}",
                    "mongo_id": None,
                    "minio_object": None,
                    "file_type": "image",
                })
        return results

    # -- Internal helpers -------------------------------------------------

    def _extract_text(
        self, file_path: str, file_type: str, case_id: str
    ) -> str:
        """Extract text from a file based on its detected type."""
        if file_type == "text":
            return extract_text_from_file(file_path)
        elif file_type == "pdf":
            return extract_text_from_pdf(file_path)
        elif file_type == "image":
            return extract_text_via_ocr(file_path, doc_id=case_id)
        else:
            logger.warning(
                "Unknown file type for '%s'. Attempting text read.", file_path
            )
            try:
                return extract_text_from_file(file_path)
            except Exception:
                return ""

    def _store_in_mongo(
        self,
        title: str,
        doc_type: str,
        case_id: str,
        source_file: str,
        text: str,
        confidence: int,
        explanation: str,
        file_type: str,
    ) -> Optional[str]:
        """Insert a document record into MongoDB. Returns the inserted ID."""
        # Normalize to POSIX separators for cross-OS consistency (B36)
        source_file = source_file.replace("\\", "/")
        doc_record = {
            "title": title,
            "doc_type": doc_type,
            "case_id": case_id,
            "source_file": source_file,
            "text": text,
            "classification_confidence": confidence,
            "classification_explanation": explanation,
            "file_type": file_type,
            "storage_backend": "local",  # default; overwritten if MinIO succeeds
            "minio_object": None,
        }
        try:
            # Dedup: skip re-ingest of same file for same case (B12)
            existing = self.mongo_collection.find_one(
                {"source_file": source_file, "case_id": case_id},
                {"_id": 1},
            )
            if existing:
                logger.info(
                    "Skipping duplicate ingest for source_file='%s' case_id='%s'; existing id=%s",
                    source_file, case_id, existing["_id"],
                )
                return existing["_id"]

            result = self.mongo_collection.insert_one(doc_record)
            logger.info(
                "Stored in MongoDB: title='%s', id=%s", title, result.inserted_id
            )
            return result.inserted_id
        except Exception as exc:
            logger.exception("MongoDB insert failed for '%s': %s", title, exc)
            return None

    def _upload_to_minio(
        self,
        file_path: str,
        mongo_id: str,
        case_id: str,
    ) -> Optional[str]:
        """Upload the raw file to MinIO. Returns the object name or None on failure.

        The object is stored at:
            {bucket}/{case_id or 'no-case'}/{mongo_id}/{filename}

        This method is non-fatal: any exception is logged as a warning and
        None is returned so ingestion continues without MinIO.
        """
        try:
            from minio import Minio

            client = Minio(
                endpoint=self._minio_endpoint,
                access_key=self._minio_access_key,
                secret_key=self._minio_secret_key,
                secure=self._minio_secure,
            )

            # Ensure bucket exists — tolerate race with concurrent worker (B29)
            if not client.bucket_exists(self._minio_bucket):
                try:
                    client.make_bucket(self._minio_bucket)
                    logger.info("Created MinIO bucket '%s'", self._minio_bucket)
                except Exception as bucket_exc:
                    # Another worker created it concurrently — safe to continue
                    logger.info(
                        "MinIO bucket '%s' already exists (concurrent create): %s",
                        self._minio_bucket, bucket_exc,
                    )

            filename = os.path.basename(file_path)
            # Sanitize case_id to prevent path traversal (B8)
            safe_case_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", case_id) if case_id else "no-case"
            object_name = f"{safe_case_id}/{mongo_id}/{filename}"

            # Stream directly to avoid doubling peak RAM (B9)
            file_size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                client.put_object(
                    bucket_name=self._minio_bucket,
                    object_name=object_name,
                    data=f,
                    length=file_size,
                )

            logger.info("Uploaded to MinIO: %s/%s", self._minio_bucket, object_name)
            return object_name

        except Exception as exc:
            logger.warning("MinIO upload failed (non-fatal): %s", exc)
            return None

    def _index_in_vectorstore(
        self,
        text: str,
        title: str,
        doc_type: str,
        case_id: str,
        source_file: str,
        mongo_id: str,
    ) -> None:
        """Split the text into chunks and add them to the Qdrant vector store.

        Each chunk carries metadata so the Case Doc RAG retriever can filter
        by case_id or doc_type.
        """
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
            return

        metadatas = [
            {
                "title": title,
                "type": doc_type,
                "case_id": case_id,
                "source_file": source_file,
                "mongo_id": mongo_id,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        try:
            self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            logger.info(
                "Indexed %d chunk(s) in vector store for '%s' (case=%s)",
                len(chunks), title, case_id,
            )
        except Exception as exc:
            logger.exception(
                "Vector store indexing failed for '%s': %s", source_file, exc
            )