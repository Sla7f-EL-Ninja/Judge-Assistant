"""
embedding_server.py
-------------------
Wraps the already-loaded BGE-M3 instance (from DocumentProcessor._get_vectorstore)
as a TEI-compatible HTTP server on port 8080.

Both MCP child processes probe http://localhost:8080 before falling back to
loading the model themselves. By answering that probe we prevent them from
loading a second (and third) copy of BGE-M3.

TEI /embed contract:
  POST /embed  {"inputs": "str | list[str]"}  → list[list[float]]
"""

import logging
import threading
from typing import Union, List

logger = logging.getLogger(__name__)


def start_embedding_server(embeddings, port: int = 8080) -> None:
    """Start a TEI-compatible embedding server in a daemon thread."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="local-embedding-server", docs_url=None, redoc_url=None)

    class EmbedRequest(BaseModel):
        inputs: Union[str, List[str]]

    @app.post("/embed")
    def embed(body: EmbedRequest):
        texts = [body.inputs] if isinstance(body.inputs, str) else body.inputs
        return embeddings.embed_documents(texts)
 
    @app.get("/health")
    def health():
        return {"status": "ok"}

    def _serve():
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )

    t = threading.Thread(target=_serve, daemon=True, name="embedding-server")
    t.start()
    logger.info("Local embedding server started on port %d", port)