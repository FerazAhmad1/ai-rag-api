from fastapi import Request
import httpx


async def get_deepseek_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.deepseek_client
