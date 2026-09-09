"""
Pure, FastAPI/DB-agnostic module (no imports from models.py, database.py,
routes/, or schemas.py, and no app.state access) - mirrors the design of
pdf_extraction.py/chunking.py. The caller (main.py's lifespan) owns the
httpx.AsyncClient's lifecycle; this module just uses one it's given.

Generates embeddings via Jina AI's hosted embeddings API.
"""

import httpx

from config import JINA_API_KEY

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
EMBEDDING_MODEL_NAME = "jina-embeddings-v5-text-small"

# Jina's v5 models only support a fixed discrete set of Matryoshka dims:
# 32, 64, 128, 256, 512, 1024 (native output is 1024-dim) - NOT an arbitrary
# range. 512 chosen per Jina's own guidance: "Matryoshka truncation to
# 256-512 dimensions is suitable for storage-constrained indexing...
# preserving strong retrieval quality above 256 dimensions."
EMBEDDING_DIMENSIONS = 512

# Jina's v5 models use task-specific adapters for asymmetric retrieval -
# Jina's own docs state this "measurably affects performance." Document
# chunks (indexed) and search questions (queries) must use different values.
TASK_RETRIEVAL_PASSAGE = "retrieval.passage"  # for document chunks, on upload
TASK_RETRIEVAL_QUERY = "retrieval.query"      # for the user's search question


def create_jina_client() -> httpx.AsyncClient:
    """Factory for the shared client - called once by main.py's lifespan.
    Not cached/singleton here; lifecycle ownership lives in main.py."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is not configured")
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        timeout=30.0,
    )


async def embed_texts(
    texts: list[str],
    task: str,
    client: httpx.AsyncClient,
) -> list[list[float]]:
    """Batch-embed texts via Jina AI, in the same order given."""
    if not texts:
        return []
    response = await client.post(
        JINA_API_URL,
        json={
            "model": EMBEDDING_MODEL_NAME,
            "input": texts,
            "task": task,
            "dimensions": EMBEDDING_DIMENSIONS,
            "embedding_type": "float",
            "normalized": True,
            "truncate": True,
        },
    )
    response.raise_for_status()
    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])

    vectors = [item["embedding"] for item in data]
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIMENSIONS}-dimensional embedding from "
                f"Jina, got {len(vector)}."
            )
    return vectors
