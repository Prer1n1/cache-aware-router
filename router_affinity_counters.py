import asyncio
import hashlib
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field


REPLICAS = [
    ("A", "http://127.0.0.1:8101"),
    ("B", "http://127.0.0.1:8102"),
]

POLICY = os.environ.get("ROUTING_POLICY", "round_robin")

if POLICY not in {"round_robin", "affinity"}:
    raise RuntimeError(f"Unsupported routing policy: {POLICY}")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "router-demo"
    messages: list[dict] = Field(min_length=1)
    stream: bool = False

    # Router metadata; never forwarded to vLLM.
    affinity_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )


def affinity_index(model: str, key: str) -> int:
    # Include the model so keys for different models remain distinct.
    data = (model + "\0" + key).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest, "big") % len(REPLICAS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.next_replica = 0
    app.state.selection_lock = asyncio.Lock()
    app.state.in_flight = {"A": 0, "B": 0}

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
    return {"status": "ok", "policy": POLICY}


@app.get("/stats")
async def stats():
    async with app.state.selection_lock:
        return {
            "policy": POLICY,
            "in_flight": dict(app.state.in_flight),
        }


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="This version supports stream=false only.",
        )

    if POLICY == "affinity" and request.affinity_key is not None:
        index = affinity_index(request.model, request.affinity_key)
        reason = "affinity"
    else:
        async with app.state.selection_lock:
            index = app.state.next_replica
            app.state.next_replica = (index + 1) % len(REPLICAS)
        reason = "round_robin"

    name, base_url = REPLICAS[index]

    # Strip router-only metadata from the model request.
    payload = request.model_dump(exclude={"affinity_key"})

    async with app.state.selection_lock:
        app.state.in_flight[name] += 1

    try:
        upstream = await app.state.client.post(
            base_url + "/v1/chat/completions",
            json=payload,
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
    finally:
        async with app.state.selection_lock:
            app.state.in_flight[name] -= 1

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get(
            "content-type", "application/json"
        ),
        headers={
            "X-Replica": name,
            "X-Routing-Policy": POLICY,
            "X-Routing-Reason": reason,
        },
    )
