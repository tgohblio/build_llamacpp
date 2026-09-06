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
  When `stream: true` is set in the job input, the handler is a generator
  that yields llama-server's SSE chunks one by one. Each yielded value is
  forwarded to the Runpod /stream endpoint as it is produced (no buffering).
  `return_aggregate_stream: True` keeps /run and /runsync working — they
  return the assembled list at completion.

If "messages" is not provided, the handler returns an error.
"""

import os
import time
import json
from typing import Iterator

import requests

import runpod

LLAMA_URL = f"http://localhost:{os.environ.get('PORT', '8080')}/v1/chat/completions"

# Defaults for non-stream path
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256

# Per-request timeout for the non-stream POST. Long: speculative decoding +
# first-call cold start can be slow.
NON_STREAM_TIMEOUT = int(os.environ.get("LLAMA_TIMEOUT", "300"))

# Per-request timeout for the stream POST. None = no read timeout; rely on
# Runpod worker idle timeout. Long generations on big contexts can run for
# tens of minutes.
STREAM_TIMEOUT_RAW = os.environ.get("LLAMA_STREAM_TIMEOUT", "").strip()
STREAM_TIMEOUT: int | None = int(STREAM_TIMEOUT_RAW) if STREAM_TIMEOUT_RAW else None

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


def llama_generate(payload: dict) -> tuple[dict, float]:
    """Forward payload to llama-server. Returns (response_json, elapsed_seconds)."""
    url = LLAMA_URL
    body = _build_body(payload)
    t0 = time.time()
    resp = requests.post(url, json=body, timeout=NON_STREAM_TIMEOUT)
    dt = time.time() - t0
    resp.raise_for_status()
    return resp.json(), dt


def llama_stream(payload: dict) -> Iterator[dict]:
    """Generator that proxies llama-server's SSE stream, one chunk per yield.

    Yields the parsed JSON chunk dict as-is (preserving OpenAI delta shape,
    including `reasoning_content` for thinking models). Terminates on the
    `data: [DONE]` sentinel. Raises on connection / HTTP errors.
    """
    url = LLAMA_URL
    body = _build_body(payload)
    body["stream"] = True

    # `requests` accepts `timeout=(connect, read)`; None disables the read
    # timeout so long generations don't get killed mid-stream.
    resp = requests.post(url, json=body, stream=True, timeout=(10, STREAM_TIMEOUT))
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        # Surface a clean error and stop — don't try to read a body that
        # may be partial / not SSE.
        raise

    for raw in resp.iter_lines():
        if not raw:
            continue
        if not raw.startswith(b"data: "):
            continue
        payload_bytes = raw[len(b"data: "):]
        if payload_bytes == b"[DONE]":
            break
        try:
            yield json.loads(payload_bytes)
        except json.JSONDecodeError:
            # Skip malformed lines rather than killing the whole stream;
            # llama-server should never emit these but be defensive.
            continue


def handler(job: dict) -> Iterator[dict] | dict:
    """Runpod job handler.

    Generator when `stream: true` is requested; single dict otherwise.
    Each yield is wrapped by the Runpod SDK as `{"output": <value>}` on
    the /stream endpoint.
    """
    job_input = job.get("input", {})

    err = _validate_input(job_input)
    if err is not None:
        return {"error": err}

    wants_stream = bool(job_input.get("stream"))

    if wants_stream:
        return _stream_handler(job_input)

    return _aggregate_handler(job_input)


def _stream_handler(job_input: dict) -> Iterator[dict]:
    """Generator path: yield llama-server chunks, then a `done` sentinel."""
    try:
        for chunk in llama_stream(job_input):
            yield chunk
    except requests.exceptions.RequestException as e:
        # Yield a final error chunk so the caller sees the failure on
        # /stream, then stop. The Runpod SDK won't buffer this — the
        # caller receives it as the last frame.
        yield {
            "error": f"llama-server stream failed: {e}",
            "choices": [],
        }
    except Exception as e:  # noqa: BLE001 — surface anything to the caller
        yield {
            "error": f"unexpected stream error: {e}",
            "choices": [],
        }
    finally:
        # Deterministic end-of-stream signal for the consumer.
        yield {"done": True}


def _aggregate_handler(job_input: dict) -> dict:
    """Non-stream path: collect the full response, return one dict."""
    try:
        result, elapsed = llama_generate(job_input)
    except requests.exceptions.RequestException as e:
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
