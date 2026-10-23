import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field


REPLICAS = [
    ("A", "http://127.0.0.1:8101"),
    ("B", "http://127.0.0.1:8102"),
]


class ChatRequest(BaseModel):
    # Preserve other model parameters, such as temperature and max_tokens.
    model_config = ConfigDict(extra="allow")

    model: str = "router-demo"
    messages: list[dict] = Field(min_length=1)
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.next_replica = 0
    app.state.selection_lock = asyncio.Lock()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=5.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
        trust_env=False,
    ) as client:
        app.state.client = client
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    # This checks the router process, not backend readiness.
    return {"status": "ok", "policy": "round_robin"}


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="This first router version supports stream=false only.",
        )

    # Protect selection, then release the lock before network I/O.
    async with app.state.selection_lock:
        index = app.state.next_replica
        app.state.next_replica = (index + 1) % len(REPLICAS)

    name, base_url = REPLICAS[index]

    try:
        upstream = await app.state.client.post(
            base_url + "/v1/chat/completions",
            json=request.model_dump(),
        )
    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail=f"Replica {name} timed out.",
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not communicate with replica {name}.",
        ) from error

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get(
            "content-type", "application/json"
        ),
        headers={"X-Replica": name},
    )
