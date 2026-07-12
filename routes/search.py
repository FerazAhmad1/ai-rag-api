from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from embeddings import embed_texts
from get_db import get_db
from models import Document, DocumentChunk
from schemas import SearchRequest, SearchResponse, SearchResultOut

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/", response_model=SearchResponse)
async def search_chunks(request: SearchRequest, db: AsyncSession = Depends(get_db)):
    query_vector = embed_texts([request.question])[0]

    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(DocumentChunk, Document.filename, distance.label("distance"))
        .join(Document, DocumentChunk.document_id == Document.id)
        .order_by(distance)
        .limit(request.top_k)
    )
    result = await db.execute(stmt)
    rows = result.all()

    return SearchResponse(
        question=request.question,
        results=[
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
        ],
    )
