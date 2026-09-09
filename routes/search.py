import httpx
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from embeddings import embed_texts, TASK_RETRIEVAL_QUERY
from get_db import get_db
from get_jina_client import get_jina_client
from models import Document, DocumentChunk
from schemas import SearchRequest, SearchResponse, SearchResultOut

router = APIRouter(prefix="/search", tags=["Search"])


async def retrieve_relevant_chunks(
    question: str,
    top_k: int,
    db: AsyncSession,
    jina_client: httpx.AsyncClient,
) -> list[SearchResultOut]:
    query_vector = (
        await embed_texts([question], task=TASK_RETRIEVAL_QUERY, client=jina_client)
    )[0]

    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(DocumentChunk, Document.filename, distance.label("distance"))
        .join(Document, DocumentChunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )
    result = await db.execute(stmt)
    rows = result.all()

    return [
        SearchResultOut(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            similarity_score=1 - distance_value,
        )
        for chunk, filename, distance_value in rows
    ]


@router.post("/", response_model=SearchResponse)
async def search_chunks(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    jina_client: httpx.AsyncClient = Depends(get_jina_client),
):
    results = await retrieve_relevant_chunks(request.question, request.top_k, db, jina_client)
    return SearchResponse(question=request.question, results=results)
