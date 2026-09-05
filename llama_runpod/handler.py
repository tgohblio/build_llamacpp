"""
Runpod serverless handler for llama.cpp with DFlash 2 speculative decoding.

llama-server is started by start_llama.sh before this script runs.
This handler forwards Runpod job input to llama-server's OpenAI-compatible
/v1/chat/completions endpoint, and returns the response as job output.

Expected job input shape (any subset is forwarded to llama-server):
{
    "messages": [{"role": "user", "content": "..."}],  # OR "prompt"
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 0.9,
    "stop": ["..."],
    "stream": false
}

If neither "messages" nor "prompt" is provided, the handler returns an error.
"""

import os
import time
import json
import requests

import runpod

LLAMA_URL = f"http://localhost:{os.environ.get('PORT', '8080')}/v1/chat/completions"
LLAMA_URL_COMPLETIONS = f"http://localhost:{os.environ.get('PORT', '8080')}/v1/completions"


def llama_generate(payload: dict) -> dict:
    """Forward payload to llama-server. Translate to chat or completions endpoint."""
    has_messages = "messages" in payload
    url = LLAMA_URL if has_messages else LLAMA_URL_COMPLETIONS

    # Build request body — only fields llama-server understands
    body = {}
    if has_messages:
        body["messages"] = payload["messages"]
    elif "prompt" in payload:
        body["prompt"] = payload["prompt"]

    for k in ("temperature", "max_tokens", "top_p", "top_k", "stop", "stream", "reasoning_effort", "presence_penalty"):
        if k in payload:
            body[k] = payload[k]

    # Defaults
    body.setdefault("temperature", 0.7)
    body.setdefault("max_tokens", 256)
    body.setdefault("top_k", 20)
    body.setdefault("min_p", 0.0)
    body.setdefault("repetition_penalty", 1.0)

    # Long timeout: speculative decoding + first-call cold start can be slow
    timeout = int(os.environ.get("LLAMA_TIMEOUT", "300"))
    t0 = time.time()
    resp = requests.post(url, json=body, timeout=timeout)
    dt = time.time() - t0
    resp.raise_for_status()
    return resp.json(), dt


def handler(job: dict) -> dict:
    """Runpod job handler."""
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        return {"error": "input must be a JSON object"}

    if "messages" not in job_input and "prompt" not in job_input:
        return {
            "error": "input must contain 'messages' (list) or 'prompt' (string)",
            "example": {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
            },
        }

    try:
        result, elapsed = llama_generate(job_input)
    except requests.exceptions.RequestException as e:
        return {"error": f"llama-server request failed: {e}"}
    except Exception as e:
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
        "return_aggregate_stream": False,
    }
)
