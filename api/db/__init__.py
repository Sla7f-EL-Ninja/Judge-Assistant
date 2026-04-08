"""
Database layer for Judge Assistant.

Modules:
- mongodb       -- MongoDB (Motor async) for cases, conversations, summaries
- qdrant        -- Qdrant vector store for RAG embeddings
- redis         -- Redis for caching, rate limiting, sessions
- minio_client  -- MinIO (S3-compatible) for file storage
- postgres      -- PostgreSQL (SQLAlchemy async) for users, roles, audit logs
- collections   -- MongoDB collection name constants
"""
