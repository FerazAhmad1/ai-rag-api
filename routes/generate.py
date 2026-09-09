import httpx
from fastapi import APIRouter, Depends

from generation import generate_answer
from get_deepseek_client import get_deepseek_client
from schemas import GenerateRequest, GenerateResponse

router = APIRouter(prefix="/generate", tags=["Generate"])

NO_CHUNKS_ANSWER = "No chunks were provided to generate an answer from."


@router.post("/", response_model=GenerateResponse)
async def generate_from_chunks(
    request: GenerateRequest,
    deepseek_client: httpx.AsyncClient = Depends(get_deepseek_client),
):
    if not request.chunks:
        return GenerateResponse(question=request.question, answer=NO_CHUNKS_ANSWER, sources=[])

    answer = await generate_answer(request.question, request.chunks, deepseek_client)
    return GenerateResponse(question=request.question, answer=answer, sources=request.chunks)
