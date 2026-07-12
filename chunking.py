"""
Pure, FastAPI/DB-agnostic module (no imports from models.py, database.py,
routes/, or schemas.py) - mirrors the design of pdf_extraction.py.

Splits per-page extracted text into token-bounded, overlapping chunks sized
against the ACTUAL tokenizer of the embedding model that will be used later
(sentence-transformers/all-MiniLM-L6-v2), not an arbitrary character count
and not the base tokenizer's generic (and here, wrong) max length.
"""

from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from pdf_extraction import PageText

# --------------------------------------------------------------------------
# Embedding-model identity & tokenizer constants
# --------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# This is the sentence-transformers wrapper's max_seq_length (defined in
# that model repo's sentence_bert_config.json), which is what actually
# governs truncation when this model is used via SentenceTransformer(...)
# to embed text.
#
# It is NOT the same as AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# .model_max_length, which reports the underlying base tokenizer's generic
# ceiling (512) and does NOT reflect where this specific model actually
# truncates. Trusting that generic value here would silently build chunks
# up to ~512 tokens that get chopped in half by the embedding step later,
# defeating the entire point of token-aware chunking.
#
# Hardcoded deliberately. Do not derive this programmatically.
MAX_SEQ_LENGTH = 256

# Content-token budget per chunk (measured with add_special_tokens=False).
# 200 content tokens + 2 special tokens ([CLS]/[SEP], added automatically by
# SentenceTransformer at embed time) = 202, comfortably under
# MAX_SEQ_LENGTH (256), leaving ~21% safety margin since chunk boundaries
# are chosen via character-based separators, not exact token boundaries.
CHUNK_SIZE_TOKENS = 200

# ~20% overlap: a common default for RAG chunking. Enough to preserve
# continuity of an idea/sentence that straddles a chunk boundary, without
# duplicating so much content that retrieval/index size suffers.
CHUNK_OVERLAP_TOKENS = 40

assert CHUNK_OVERLAP_TOKENS < CHUNK_SIZE_TOKENS, "overlap must be smaller than chunk size"


@lru_cache(maxsize=1)
def _get_tokenizer() -> PreTrainedTokenizerBase:
    """
    Lazily load and cache the tokenizer for EMBEDDING_MODEL_NAME.

    First call in a process may trigger a small (few hundred KB) one-time
    download from Hugging Face Hub into ~/.cache/huggingface if this model
    isn't already cached locally. Subsequent calls (in this or later
    processes, once cached on disk) are instant and offline.
    """
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def _count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Content-token count only (add_special_tokens=False) - matches how
    CHUNK_SIZE_TOKENS is budgeted, i.e. NOT what the model sees after
    [CLS]/[SEP] insertion at embedding time."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_index: int  # 0-based, sequential across the whole document
    page_number: int  # 1-indexed, matches PageText.page_number
    text: str
    token_count: int  # content tokens only (add_special_tokens=False)
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
    tokenizer: PreTrainedTokenizerBase,
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
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pieces = []
    for start in range(0, len(input_ids), token_budget):
        token_slice = input_ids[start : start + token_budget]
        pieces.append(tokenizer.decode(token_slice, skip_special_tokens=True))
    return pieces


def _split_recursive(
    text: str,
    separators: list[str],
    tokenizer: PreTrainedTokenizerBase,
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
    tokenizer: PreTrainedTokenizerBase,
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
    tokenizer: PreTrainedTokenizerBase,
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
