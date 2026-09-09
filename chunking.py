"""
Pure, FastAPI/DB-agnostic module (no imports from models.py, database.py,
routes/, or schemas.py) - mirrors the design of pdf_extraction.py.

Splits per-page extracted text into token-bounded, overlapping chunks. Token
counts are measured with tiktoken (OpenAI's tokenizer) as an approximation -
the embedding backend is Jina AI (embeddings.py), which uses its own,
different tokenizer. This is fine here: token counts only drive a
retrieval-granularity chunk-size choice (~200 tokens), not truncation
avoidance - Jina's real context window (32K tokens) is so much larger than
our budget that approximate counts can't meaningfully risk truncation.
Loading Jina's actual tokenizer would require the ~104MB `transformers`
package for no real benefit at this scale, so tiktoken stays.
"""

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from pdf_extraction import PageText

# --------------------------------------------------------------------------
# Embedding-model identity & tokenizer constants
# --------------------------------------------------------------------------

# tiktoken's cl100k_base encoding - an approximation for chunk sizing, not
# Jina's exact tokenizer (see module docstring for why that's fine here).
TIKTOKEN_ENCODING_NAME = "cl100k_base"

# Jina's actual context window for jina-embeddings-v5-text-small is 32000
# tokens - vastly larger than the 200-token budget below. Kept here as a
# documented ceiling, not something CHUNK_SIZE_TOKENS is pushed close to.
# This constant is NOT referenced anywhere else in this file (verified) -
# actual chunk sizing is driven entirely by CHUNK_SIZE_TOKENS/
# CHUNK_OVERLAP_TOKENS below, which are unchanged from local-model tuning.
MAX_SEQ_LENGTH = 32000

# Content-token budget per chunk. This is now a RETRIEVAL GRANULARITY choice,
# not a truncation-avoidance one (the model's real limit is 8191, ~41x this
# budget) - smaller, focused chunks retrieve more precisely than large ones.
# Kept at the same value tuned/verified when the embedding backend was local.
CHUNK_SIZE_TOKENS = 200

# ~20% overlap: a common default for RAG chunking. Enough to preserve
# continuity of an idea/sentence that straddles a chunk boundary, without
# duplicating so much content that retrieval/index size suffers.
CHUNK_OVERLAP_TOKENS = 40

assert CHUNK_OVERLAP_TOKENS < CHUNK_SIZE_TOKENS, "overlap must be smaller than chunk size"


@lru_cache(maxsize=1)
def _get_tokenizer() -> tiktoken.Encoding:
    """
    Lazily load and cache the tiktoken encoding.

    First call in a process may trigger a small one-time download of the
    BPE ranks file (cached locally afterward, typically under
    ~/.cache/tiktoken or TIKTOKEN_CACHE_DIR if set). Subsequent calls are
    instant and offline.
    """
    return tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)


def _count_tokens(text: str, tokenizer: tiktoken.Encoding) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_index: int  # 0-based, sequential across the whole document
    page_number: int  # 1-indexed, matches PageText.page_number
    text: str
    token_count: int  # tiktoken (cl100k_base) token count
    char_count: int


# --------------------------------------------------------------------------
# Recursive, structure-aware splitting
# --------------------------------------------------------------------------

# Tried in priority order: paragraph -> line -> sentence-ish -> word.
# Raw token-level fallback is handled separately (see _hard_slice_by_tokens)
# since it needs token-level, not character-level, slicing.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]


def _split_text_by_separator(text: str, separator: str) -> list[str]:
    """
    Split `text` on `separator`, re-attaching the separator to the end of
    each piece (except the trailing piece) so paragraph/sentence punctuation
    is preserved when pieces are later rejoined into chunks.
    """
    parts = text.split(separator)
    pieces = [p + separator for p in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return [p for p in pieces if p]


def _hard_slice_by_tokens(
    text: str,
    tokenizer: tiktoken.Encoding,
    token_budget: int,
) -> list[str]:
    """
    Last-resort fallback for a single "word" (no separators at all - e.g. a
    very long URL or hash string) that still exceeds token_budget after
    trying every separator in _SEPARATORS.

    Slices at the token level using the tokenizer's own ids, guaranteeing
    every returned piece is <= token_budget tokens. This can produce minor
    whitespace/detokenization artifacts on decode (acceptable - this path
    only triggers on pathological, non-prose input).
    """
    input_ids = tokenizer.encode(text)
    pieces = []
    for start in range(0, len(input_ids), token_budget):
        token_slice = input_ids[start : start + token_budget]
        pieces.append(tokenizer.decode(token_slice))
    return pieces


def _split_recursive(
    text: str,
    separators: list[str],
    tokenizer: tiktoken.Encoding,
    token_budget: int,
) -> list[str]:
    """
    Recursively split `text` using the separator hierarchy until every
    returned piece is <= token_budget tokens. Falls back to hard token-level
    slicing once separators are exhausted.
    """
    if _count_tokens(text, tokenizer) <= token_budget:
        return [text] if text else []

    if not separators:
        return _hard_slice_by_tokens(text, tokenizer, token_budget)

    separator, *rest_separators = separators
    pieces = _split_text_by_separator(text, separator)

    atomic_units: list[str] = []
    for piece in pieces:
        if _count_tokens(piece, tokenizer) <= token_budget:
            atomic_units.append(piece)
        else:
            atomic_units.extend(
                _split_recursive(piece, rest_separators, tokenizer, token_budget)
            )
    return atomic_units


# --------------------------------------------------------------------------
# Greedy packing with overlap
# --------------------------------------------------------------------------

def _carry_over_overlap(
    units: list[str],
    tokenizer: tiktoken.Encoding,
    chunk_overlap_tokens: int,
) -> list[str]:
    """Take trailing units from `units` (a just-finished chunk's pieces)
    summing to ~chunk_overlap_tokens, to seed the start of the next chunk."""
    if chunk_overlap_tokens <= 0:
        return []

    overlap_units: list[str] = []
    overlap_tokens = 0
    for unit in reversed(units):
        if overlap_tokens >= chunk_overlap_tokens:
            break
        overlap_units.insert(0, unit)
        overlap_tokens += _count_tokens(unit, tokenizer)
    return overlap_units


def _pack_atomic_units(
    atomic_units: list[str],
    tokenizer: tiktoken.Encoding,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[str]:
    """
    Greedily merge atomic_units (each already <= chunk_size_tokens) into
    chunk texts up to chunk_size_tokens, carrying over ~chunk_overlap_tokens
    of trailing units from the previous chunk into the next.
    """
    if not atomic_units:
        return []

    chunk_texts: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in atomic_units:
        unit_tokens = _count_tokens(unit, tokenizer)

        if current_units and current_tokens + unit_tokens > chunk_size_tokens:
            chunk_texts.append("".join(current_units))
            current_units = _carry_over_overlap(current_units, tokenizer, chunk_overlap_tokens)
            current_tokens = sum(_count_tokens(u, tokenizer) for u in current_units)

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunk_texts.append("".join(current_units))

    return chunk_texts


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def chunk_pages(
    pages: list[PageText],
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Chunk each page's text independently (chunks never span a page
    boundary - see Chunk.page_number, a single int per the current schema).
    chunk_index is 0-based and sequential across the whole document.

    Pages with no extractable text (e.g. scanned/image-only pages) yield
    zero chunks for that page - this is not an error.
    """
    tokenizer = _get_tokenizer()
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        atomic_units = _split_recursive(text, _SEPARATORS, tokenizer, chunk_size_tokens)
        packed = _pack_atomic_units(atomic_units, tokenizer, chunk_size_tokens, chunk_overlap_tokens)

        for chunk_text in packed:
            stripped = chunk_text.strip()
            if not stripped:
                continue
            chunks.append(
                Chunk(
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    text=stripped,
                    token_count=_count_tokens(stripped, tokenizer),
                    char_count=len(stripped),
                )
            )
            chunk_index += 1

    return chunks
