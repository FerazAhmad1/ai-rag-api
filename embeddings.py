"""
Pure, FastAPI/DB-agnostic module (no imports from models.py, database.py,
routes/, or schemas.py) - mirrors the lazy-loading pattern in chunking.py.

Generates embeddings with the same model chunking.py sizes chunks against,
so document and query embeddings always come from one consistent model.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from chunking import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Lazily load and cache the embedding model.

    First call in a process downloads the model weights (~90MB) from
    Hugging Face Hub into ~/.cache/huggingface if not already cached.
    Subsequent calls (in this or later processes, once cached) are instant.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-encode texts into embedding vectors, in the same order given."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()
