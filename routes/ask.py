import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from generation import generate_answer
from get_db import get_db
from get_deepseek_client import get_deepseek_client
from get_jina_client import get_jina_client
from routes.search import retrieve_relevant_chunks
from schemas import AskResponse, SearchRequest

router = APIRouter(prefix="/ask", tags=["Ask"])

NO_CONTEXT_ANSWER = (
    "I don't have any relevant documents to answer that question. "
    "Try uploading a document first, or rephrasing your question."
)


@router.post("/", response_model=AskResponse)
async def ask_question(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    jina_client: httpx.AsyncClient = Depends(get_jina_client),
    deepseek_client: httpx.AsyncClient = Depends(get_deepseek_client),
):
    sources = await retrieve_relevant_chunks(request.question, request.top_k, db, jina_client)

    if not sources:
        return AskResponse(question=request.question, answer=NO_CONTEXT_ANSWER, sources=[])

    answer = await generate_answer(request.question, sources, deepseek_client)
    return AskResponse(question=request.question, answer=answer, sources=sources)
