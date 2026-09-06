"""OpenAI-compatible HTTP proxy for llama-server.

Exposes a minimal OpenAI-style surface so the OpenAI Python / Node SDKs can
talk to llama-server directly via the Runpod serverless container.

Endpoints:
  GET  /health                       — proxy liveness, used by start_llama.sh
  GET  /v1/models                    — list the configured model(s)
  POST /v1/chat/completions          — chat (streaming + non-streaming)
  POST /v1/completions               — legacy text completions (streaming + non-streaming)

The proxy forwards every other field of the request body to llama-server
unchanged. Streaming is performed with httpx.AsyncClient (no read timeout
by default) and a FastAPI StreamingResponse emitting proper SSE framing
(`data: {json}\n\n` + a final `data: [DONE]\n\n`).

Run:
  python3 -m uvicorn oai_proxy:app --host 0.0.0.0 --port ${OAI_PORT:-8081}

Env:
  PORT        — llama-server port (default 8080)
  OAI_PORT    — listen port for this proxy (default 8081)
  MODEL       — model id surfaced by /v1/models (e.g. "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M")
  OAI_TIMEOUT — connect timeout in seconds for the proxy→llama-server hop
                (default 10). Read timeout is None unless OAI_READ_TIMEOUT is set.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

LOG = logging.getLogger("oai_proxy")
logging.basicConfig(level=os.environ.get("OAI_LOG_LEVEL", "INFO"))

LLAMA_HOST = f"http://localhost:{os.environ.get('PORT', '8080')}"
LLAMA_CHAT = f"{LLAMA_HOST}/v1/chat/completions"
LLAMA_COMPLETIONS = f"{LLAMA_HOST}/v1/completions"

OAI_PORT = int(os.environ.get("OAI_PORT", "8081"))
MODEL_ID = os.environ.get("MODEL", "llama-cpp")

CONNECT_TIMEOUT = float(os.environ.get("OAI_TIMEOUT", "10"))
READ_TIMEOUT_RAW = os.environ.get("OAI_READ_TIMEOUT", "").strip()
READ_TIMEOUT: httpx.Timeout | float | None
if READ_TIMEOUT_RAW:
    READ_TIMEOUT = float(READ_TIMEOUT_RAW)
else:
    # None read timeout: long generations on big contexts can run for tens
    # of minutes; rely on uvicorn's keep-alive / Runpod worker idle timeout.
    READ_TIMEOUT = httpx.Timeout(CONNECT_TIMEOUT, read=None)

app = FastAPI(title="llama.cpp OpenAI-compatible proxy", version="1.0.0")


# --- Health ------------------------------------------------------------------
@app.get("/health")
async def health() -> PlainTextResponse:
    """Proxy liveness. start_llama.sh polls this until the sidecar is up."""
    return PlainTextResponse("ok")


# --- /v1/models --------------------------------------------------------------
@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Stub model listing. Only the configured MODEL is surfaced."""
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }
    )


# --- /v1/chat/completions + /v1/completions ---------------------------------
async def _iter_sse(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Yield properly framed SSE bytes from an upstream httpx streaming response.

    llama-server already emits `data: {json}\n\n` + a terminating
    `data: [DONE]\n\n`, so we just forward bytes verbatim. We add a defensive
    newline in case the upstream omits one before [DONE].
    """
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        if not line.startswith(b"data: "):
            # Some upstreams send a bare `data:{json}` without a space after
            # the colon. Normalize.
            if line.startswith(b"data:"):
                line = b"data: " + line[len(b"data:"):]
            else:
                continue
        yield line + b"\n\n"
        if line == b"data: [DONE]":
            break


async def _forward_streaming(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> StreamingResponse:
    """Stream llama-server's response back as proper SSE."""
    body = {**body, "stream": True}
    upstream = await client.send(
        client.build_request("POST", url, json=body),
        stream=True,
    )
    if upstream.status_code >= 400:
        # Drain to release the connection, then return a JSON error.
        err_body = await upstream.aread()
        await upstream.aclose()
        try:
            err_payload: Any = json.loads(err_body)
        except json.JSONDecodeError:
            err_payload = {"error": err_body.decode("utf-8", errors="replace")}
        return JSONResponse(err_payload, status_code=upstream.status_code)

    async def gen() -> AsyncIterator[bytes]:
        try:
            async for chunk in _iter_sse(upstream):
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx output buffering
            "Connection": "keep-alive",
        },
    )


async def _forward_non_streaming(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> JSONResponse:
    """Forward non-streaming request and return llama-server's JSON as-is."""
    body = {**body, "stream": False}
    upstream = await client.post(url, json=body)
    # Prefer upstream.json() (httpx handles decoding); fall back to raw
    # bytes if the response isn't valid JSON (e.g. 502 from an LB).
    try:
        payload: Any = upstream.json()
    except (json.JSONDecodeError, ValueError):
        payload = {"error": upstream.text}
    return JSONResponse(
        content=payload,
        status_code=upstream.status_code,
        headers={"Content-Type": "application/json"},
    )


async def _proxy(request: Request, url: str) -> JSONResponse | StreamingResponse:
    """Common path for /v1/chat/completions and /v1/completions."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": f"invalid JSON body: {e}"}, status_code=400
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "request body must be a JSON object"}, status_code=400
        )

    wants_stream = bool(body.get("stream"))
    async with httpx.AsyncClient(timeout=READ_TIMEOUT) as client:
        if wants_stream:
            return await _forward_streaming(client, url, body)
        return await _forward_non_streaming(client, url, body)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy(request, LLAMA_CHAT)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy(request, LLAMA_COMPLETIONS)


# --- Local runner (so we can smoke-test without uvicorn) --------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "oai_proxy:app",
        host="0.0.0.0",
        port=OAI_PORT,
        log_level=os.environ.get("OAI_LOG_LEVEL", "info").lower(),
    )
