from fastapi import Request
import httpx


async def get_jina_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.jina_client
