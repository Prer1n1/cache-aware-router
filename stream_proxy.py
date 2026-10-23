"""Forward a backend stream and release its resources on completion."""

import anyio
import httpx
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse


async def release(app, name, upstream=None):
    # Protect cleanup against cancellation during client disconnects.
    with anyio.CancelScope(shield=True):
        try:
            if upstream is not None:
                await upstream.aclose()
        finally:
            async with app.state.selection_lock:
                app.state.in_flight[name] -= 1


class ManagedStream(StreamingResponse):
    def __init__(self, app, name, upstream, headers):
        self.router_app = app
        self.replica_name = name
        self.upstream = upstream

        super().__init__(
            content=upstream.aiter_bytes(),
            status_code=upstream.status_code,
            media_type="text/event-stream",
            headers={
                **headers,
                "Cache-Control": "no-cache",
            },
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await release(
                self.router_app,
                self.replica_name,
                self.upstream,
            )


async def forward_stream(app, name, base_url, payload, headers):
    # The caller has already incremented this replica's counter.
    try:
        outgoing = app.state.client.build_request(
            "POST",
            base_url + "/v1/chat/completions",
            json=payload,
        )
        upstream = await app.state.client.send(
            outgoing,
            stream=True,
        )
    except httpx.TimeoutException as error:
        await release(app, name)
        raise HTTPException(
            status_code=504,
            detail=f"Replica {name} timed out before streaming.",
        ) from error
    except httpx.RequestError as error:
        await release(app, name)
        raise HTTPException(
            status_code=502,
            detail=f"Could not communicate with replica {name}.",
        ) from error
    except BaseException:
        await release(app, name)
        raise

    # Preserve backend error responses before starting a successful stream.
    if upstream.status_code != 200:
        try:
            body = await upstream.aread()
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type=upstream.headers.get(
                    "content-type", "application/json"
                ),
                headers=headers,
            )
        finally:
            await release(app, name, upstream)

    try:
        return ManagedStream(app, name, upstream, headers)
    except BaseException:
        await release(app, name, upstream)
        raise
