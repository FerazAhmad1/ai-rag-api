"""
Pure, FastAPI/DB-agnostic module - mirrors embeddings.py's design. Generates
a grounded answer via DeepSeek's chat completions API from retrieved chunks
+ a question. The caller (main.py's lifespan) owns the httpx.AsyncClient's
lifecycle; this module just uses one it's given.
"""

import httpx

from config import DEEPSEEK_API_KEY

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
GENERATION_MODEL_NAME = "deepseek-v4-flash"

TEMPERATURE = 0.2
MAX_TOKENS = 500


SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "Do not use outside knowledge or information from your training data. "
    "You may combine, summarize, and reason over information present in the "
    "context, but every factual claim must be supported by the context. "

    "Identify the information that is relevant to the user's question and "
    "omit unrelated information. Preserve important names, keywords, terms, "
    "facts, and values from the context when they are relevant to the answer. "

    "When multiple entities match the question, include the relevant matches "
    "and order them from most relevant to least relevant. "

    "Present the answer in a concise, structured format that is appropriate "
    "for the question. Use short labels such as Role, Skills, Experience, "
    "Source, or other relevant labels when they improve clarity. Do not "
    "include a category merely because the information exists in the context. "

    "Do not copy entire context chunks or reproduce unrelated information. "
    "Do not invent facts, details, categories, or values. "

    "When the context does not contain enough information to answer the "
    "question, respond only with: "
    "'the information is not available in the uploaded documents.'"
)

def create_deepseek_client() -> httpx.AsyncClient:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        timeout=60.0,
    )


def _build_context_block(sources: list) -> str:
    parts = [
        f"[Source: {s.filename}, page {s.page_number}]\n{s.content}" for s in sources
    ]
    return "\n\n".join(parts)


async def generate_answer(
    question: str,
    sources: list,
    client: httpx.AsyncClient,
) -> str:
    """sources: list[SearchResultOut] (or anything with .filename/.page_number/.content)."""
    context_block = _build_context_block(sources)
    user_content = f"Context:\n\n{context_block}\n\nQuestion: {question}"

    response = await client.post(
        DEEPSEEK_API_URL,
        json={
            "model": GENERATION_MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
