"""
Runpod serverless handler for llama.cpp with DFlash 2 speculative decoding.

llama-server is started by start_llama.sh before this script runs.
This handler receives jobs from the Runpod serverless API and forwards them
to llama-server's OpenAI-compatible /v1/chat/completions endpoint.

Job input shape (set as the `input` field in the Runpod API request body):
{
    "input": {
        "messages": [{"role": "user", "content": "..."}],
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.9,
        "stop": ["..."],
        "stream": false,
        "stream_options": {"include_usage": true}
    }
}

Streaming:
  When `stream: true` is set in the job input, the handler is an async
  generator that yields llama-server's SSE chunks one by one. Each yielded
  value is forwarded to the Runpod /stream endpoint as it is produced.
  `return_aggregate_stream: True` keeps /run and /runsync working — they
  return the assembled list at completion.

If "messages" is not provided, the handler returns an error.
"""

import os
import time
import json
from typing import AsyncIterator

import aiohttp

import runpod

LLAMA_URL = f"http://localhost:{os.environ.get('PORT', '8080')}/v1/chat/completions"

# Defaults for non-stream path
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256

# Per-request timeout for the non-stream POST. Long: speculative decoding +
# first-call cold start can be slow.
NON_STREAM_TIMEOUT = int(os.environ.get("LLAMA_TIMEOUT", "300"))

# Per-request timeout for the stream POST. Defaults to 600s; set to 0 to
# disable (rely on Runpod worker idle timeout). Long generations on big
# contexts can run for tens of minutes so we use a sensible default rather
# than None (indefinite).
STREAM_TIMEOUT_RAW = os.environ.get("LLAMA_STREAM_TIMEOUT", "").strip()
if STREAM_TIMEOUT_RAW:
    STREAM_TIMEOUT: int | None = int(STREAM_TIMEOUT_RAW)
else:
    STREAM_TIMEOUT = 600  # 10 minutes default

# Fields we forward to llama-server. Kept narrow to avoid passing unknowns.
_FORWARD_FIELDS = (
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "stream",
    "stream_options",
    "frequency_penalty",
    "presence_penalty",
)


def _build_body(payload: dict) -> dict:
    """Translate a Runpod job input into a llama-server request body."""
    body: dict = {"messages": payload["messages"]}

    for k in _FORWARD_FIELDS:
        if k in payload:
            body[k] = payload[k]

    # Defaults only when not streaming; llama-server treats an omitted
    # max_tokens as "no limit", which we don't want for non-stream callers.
    if not body.get("stream"):
        body.setdefault("temperature", DEFAULT_TEMPERATURE)
        body.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    return body


def _validate_input(job_input: dict) -> str | None:
    """Return an error string if the job input is malformed, else None."""
    if not isinstance(job_input, dict):
        return "input must be a JSON object"
    if "messages" not in job_input:
        return (
            "input must contain 'messages' (list); "
            "example: {\"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}"
        )
    return None


async def llama_generate(payload: dict) -> tuple[dict, float]:
    """Forward payload to llama-server. Returns (response_json, elapsed_seconds)."""
    url = LLAMA_URL
    body = _build_body(payload)
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=NON_STREAM_TIMEOUT)) as resp:
            resp.raise_for_status()
            result = await resp.json()
    dt = time.time() - t0
    return result, dt


async def llama_stream(payload: dict) -> AsyncIterator[dict]:
    """Async generator that proxies llama-server's SSE stream, one chunk per yield.

    Yields the parsed JSON chunk dict as-is (preserving OpenAI delta shape,
    including `reasoning_content` for thinking models). Terminates on the
    `data: [DONE]` sentinel. Raises on connection / HTTP errors.
    """
    url = LLAMA_URL
    body = _build_body(payload)
    body["stream"] = True

    timeout = aiohttp.ClientTimeout(total=STREAM_TIMEOUT)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, timeout=timeout) as resp:
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError:
                # Surface a clean error and stop — don't try to read a body that
                # may be partial / not SSE.
                raise

            async for raw in resp.content:
                raw = raw.decode("utf-8")
                if not raw.strip():
                    continue
                if not raw.startswith("data: "):
                    continue
                payload_str = raw[len("data: "):].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    yield json.loads(payload_str)
                except json.JSONDecodeError:
                    # Skip malformed lines rather than killing the whole stream.
                    continue


async def handler(job: dict) -> AsyncIterator[dict] | dict:
    """Runpod job handler — async generator when streaming, single dict otherwise.

    The async generator shape is what makes `is_asyncgenfunction(handler)` return
    True, which routes to `run_job_generator()` in the SDK. Each `yield` is
    wrapped by the SDK as `{"output": <value>}` on the /stream endpoint.
    """
    job_input = job.get("input", {})

    err = _validate_input(job_input)
    if err is not None:
        return {"error": err}

    wants_stream = bool(job_input.get("stream"))

    if wants_stream:
        async for chunk in _stream_handler(job_input):
            yield chunk
        return  # unreachable but explicit

    return await _aggregate_handler(job_input)


async def _stream_handler(job_input: dict) -> AsyncIterator[dict]:
    """Async generator path: yield llama-server chunks, then a `done` sentinel."""
    try:
        async for chunk in llama_stream(job_input):
            yield chunk
    except aiohttp.ClientError as e:
        yield {
            "error": f"llama-server stream failed: {e}",
            "choices": [],
        }
    except Exception as e:  # noqa: BLE001
        yield {
            "error": f"unexpected stream error: {e}",
            "choices": [],
        }
    finally:
        yield {"done": True}


async def _aggregate_handler(job_input: dict) -> dict:
    """Non-stream path: collect the full response, return one dict."""
    try:
        result, elapsed = await llama_generate(job_input)
    except aiohttp.ClientError as e:
        return {"error": f"llama-server request failed: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"unexpected error: {e}"}

    return {
        "response": result,
        "elapsed_seconds": round(elapsed, 3),
    }


# Health endpoint: /ping returns llama-server /health status
def health_check() -> dict:
    try:
        r = requests.get(f"http://localhost:{os.environ.get('PORT', '8080')}/health", timeout=5)
        r.raise_for_status()
        return {"status": "healthy", "llama": r.json()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Custom /ping handler for runpod serverless health checks
def _ping_handler(*args, **kwargs):
    return health_check()


# Start the Runpod serverless worker
runpod.serverless.start(
    {
        "handler": handler,
        "ping": _ping_handler,
        "return_aggregate_stream": True,
    }
)
