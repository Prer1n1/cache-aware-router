import asyncio
import hashlib
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from stream_proxy import forward_stream


REPLICAS = [
    ("A", "http://127.0.0.1:8101"),
    ("B", "http://127.0.0.1:8102"),
]

POLICY = os.environ.get("ROUTING_POLICY", "round_robin")
LOAD_THRESHOLD = int(os.environ.get("LOAD_THRESHOLD", "2"))

if POLICY not in {"round_robin", "least_in_flight", "affinity", "affinity_load"}:
    raise RuntimeError(f"Unsupported policy: {POLICY}")

if LOAD_THRESHOLD < 1:
    raise RuntimeError("LOAD_THRESHOLD must be at least 1")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "router-demo"
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    affinity_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )


def affinity_index(model: str, key: str) -> int:
    data = (model + "\0" + key).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest, "big") % len(REPLICAS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.next_replica = 0
    app.state.selection_lock = asyncio.Lock()
    app.state.in_flight = {name: 0 for name, _ in REPLICAS}

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
            "load_threshold": LOAD_THRESHOLD,
            "in_flight": dict(app.state.in_flight),
        }


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    payload = request.model_dump(exclude={"affinity_key"})

    # Select and reserve a destination in one atomic critical section.
    async with app.state.selection_lock:
        if POLICY == "least_in_flight":
            minimum = min(app.state.in_flight.values())

            # Rotate preference when replicas have equal counts.
            start = app.state.next_replica
            candidates = [
                (start + offset) % len(REPLICAS)
                for offset in range(len(REPLICAS))
            ]

            index = next(
                i for i in candidates
                if app.state.in_flight[REPLICAS[i][0]] == minimum
            )

            app.state.next_replica = (index + 1) % len(REPLICAS)
            reason = "least_in_flight"

        elif POLICY != "round_robin" and request.affinity_key is not None:
            preferred = affinity_index(
                request.model, request.affinity_key
            )
            index = preferred
            reason = "affinity"

            if POLICY == "affinity_load":
                least_loaded = min(
                    range(len(REPLICAS)),
                    key=lambda i: app.state.in_flight[REPLICAS[i][0]],
                )

                preferred_load = app.state.in_flight[
                    REPLICAS[preferred][0]
                ]
                minimum_load = app.state.in_flight[
                    REPLICAS[least_loaded][0]
                ]

                if preferred_load - minimum_load >= LOAD_THRESHOLD:
                    index = least_loaded
                    reason = "load_override"
        else:
            index = app.state.next_replica
            app.state.next_replica = (index + 1) % len(REPLICAS)
            reason = "round_robin"

        name, base_url = REPLICAS[index]
        app.state.in_flight[name] += 1

    if request.stream:
        return await forward_stream(
            app, name, base_url, payload,
            headers={
                "X-Replica": name,
                "X-Routing-Policy": POLICY,
                "X-Routing-Reason": reason,
            },
        )

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
