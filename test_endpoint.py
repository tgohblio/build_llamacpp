"""
Client for calling a Runpod serverless endpoint via the Runpod SDK.

Calls go through the Runpod serverless API (run_sync / run / stream),
NOT the OpenAI-compatible proxy URL.

Usage:
    # Synchronous — waits for completion, returns full response
    python llama_runpod/test_endpoint.py "Your prompt here"

    # Streaming — submits async job, streams chunks via /stream
    python llama_runpod/test_endpoint.py --stream "Write a 5 sentence poem about rainbow"

    # Async with status polling
    python llama_runpod/test_endpoint.py --async "Your prompt here"

Environment variables:
    RUNPOD_API_KEY     — your Runpod API key (required)
    RUNPOD_ENDPOINT_ID — your endpoint ID

Requirements: runpod>=1.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time as time_module
from dotenv import load_dotenv

import runpod
from runpod.http_client import AsyncClientSession
from runpod.endpoint.asyncio import Endpoint as AsyncEndpoint

load_dotenv()
api_key = os.environ.get("RUNPOD_API_KEY")
endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")

def get_endpoint() -> runpod.Endpoint:
    """Create a Runpod SDK endpoint object."""
    if not api_key:
        raise ValueError("RUNPOD_API_KEY environment variable is required")
    return runpod.Endpoint(str(endpoint_id), api_key)


def build_input(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict:
    """Build the job input payload forwarded to llama-server."""
    return {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "reasoning_effort": "low"
    }


def _chat_content(output: dict) -> str | None:
    """Extract text from a llama-server chat completions response dict.

    The handler returns: {"elapsed_seconds": N, "response": {OpenAI chat completion}}
    So run_sync's output field is: {"elapsed_seconds": N, "response": {...}}

    The response dict can have two shapes:
      - Chat completions: {"choices": [{"message": {"content": "..."}}]}
      - Text completions:  {"text": "..."} or {"text": ["..."]}
    """
    inner = output.get("response", output)

    # Chat completions shape: choices[0].message.content
    if "choices" in inner:
        choice = inner["choices"][0]
        if "message" in choice:
            return choice["message"].get("content")
        if "text" in choice:
            return choice["text"]
    # Text completions shape
    if "text" in inner:
        text = inner["text"]
        return "".join(text) if isinstance(text, list) else text
    return None


def run_sync(
    endpoint: runpod.Endpoint,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout: int = 120,
) -> None:
    """Submit a synchronous job via /runsync. Blocks until completion."""
    job_input = build_input(messages, temperature, max_tokens, stream=False)

    result: dict | list = endpoint.run_sync(job_input, timeout=timeout)

    # Unwrap list from return_aggregate_stream
    if isinstance(result, list):
        result = result[0] if result else {}

    text = _chat_content(result)
    if text:
        print(text)
    else:
        print(result)


def run_async(
    endpoint: runpod.Endpoint,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout: int = 300,
) -> None:
    """Submit an async job via /run, poll status, print output on completion."""
    job_input = build_input(messages, temperature, max_tokens, stream=False)

    run_request = endpoint.run(job_input)
    job_id = run_request.job_id

    status = run_request.status()
    if status != "COMPLETED":
        output = run_request.output(timeout=timeout)
    else:
        output = run_request.output()

    # Unwrap list from return_aggregate_stream
    if isinstance(output, list):
        output = output[0] if output else {}

    if isinstance(output, dict):
        text = _chat_content(output)
        if text:
            print(text)
        else:
            print(f"[run] job {status} (no content extracted): {output}", file=sys.stderr)
            sys.exit(1)
    elif isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                text = _chat_content(item)
                if text:
                    print(text)
                    return
        print(f"[run] job {status} (no content extracted): {output}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[run] job {status}: {output}", file=sys.stderr)
        sys.exit(1)


def run_streaming(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stream_timeout: int | None = None,
) -> None:
    """Submit an async job and stream chunks via /stream (async SDK, no 10s timeout).

    Args:
        stream_timeout: Max seconds to wait for streaming to complete.
            Defaults to the LLAMA_STREAM_TIMEOUT env var (120s if unset).
            None disables the client-side timeout (relies on handler timeout).
    """
    if stream_timeout is None:
        stream_timeout = int(os.environ.get("LLAMA_STREAM_TIMEOUT") or "120")

    asyncio.run(_run_streaming_async(
        messages, temperature, max_tokens, stream_timeout
    ))


async def _run_streaming_async(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stream_timeout: int | None,
) -> None:
    """Async implementation — uses aiohttp with 600s poll timeout instead of 10s."""
    async with AsyncClientSession() as session:
        aio_endpoint = AsyncEndpoint(endpoint_id, session, api_key)
        job_input = build_input(messages, temperature, max_tokens, stream=True)

        print("--- Streaming response ---", file=sys.stderr)
        job = await aio_endpoint.run(job_input)
        start = time_module.time()

        # stream() is an async generator that polls /stream/{job_id} with
        # aiohttp's 600s timeout — no 10s hard limit like the sync SDK.
        # Each frame is the dict yielded by the handler's async generator directly
        # (i.e. the parsed SSE chunk from llama-server).
        async for frame in job.stream():
            elapsed = time_module.time() - start
            if stream_timeout and elapsed > stream_timeout:
                print(f"\n[stream] client timeout after {elapsed:.0f}s", file=sys.stderr)
                break

            if not isinstance(frame, dict):
                continue
            output: dict = frame

            # Propagate errors from the handler
            if "error" in output:
                print(f"\n[stream] ERROR: {output['error']}", file=sys.stderr)
                break

            # Handler sends a {"done": True} sentinel when the stream finishes.
            if output.get("done"):
                break

            # OpenAI chat streaming: choices[0].delta.content
            if "choices" in output:
                delta = output["choices"][0].get("delta", {})
                # Only show actual response content — skip reasoning_content
                # (Qwen3 thinking models emit thinking via reasoning_content first,
                # then actual response via content in later chunks).
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
            # Fallback: plain text (e.g. /v1/completions)
            elif "text" in output:
                text = output["text"]
                if isinstance(text, list):
                    text = "".join(text)
                print(text, end="", flush=True)
        print("\n--- End stream ---", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call a Runpod llama.cpp serverless endpoint via the Runpod SDK"
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text")
    parser.add_argument(
        "--chat", nargs=2, metavar=("SYSTEM", "USER"),
        help="Chat mode: <system> <user>"
    )
    parser.add_argument("--stream", action="store_true",
                        help="Stream chunks via /stream (async job)")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="Async job via /run with status polling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout in seconds for /runsync (default: 120)"
    )
    parser.add_argument(
        "--stream-timeout", type=int, default=None,
        help="Client-side streaming timeout in seconds. "
             "Defaults to LLAMA_STREAM_TIMEOUT env var (120s if unset)."
    )

    args = parser.parse_args()

    if not args.prompt and not args.chat:
        parser.print_help()
        sys.exit(1)

    if args.chat:
        system, user = args.chat
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    else:
        messages = [{"role": "user", "content": args.prompt}]

    endpoint = get_endpoint()

    if args.stream:
        run_streaming(messages, args.temperature, args.max_tokens,
                       stream_timeout=args.stream_timeout)
    elif args.async_mode:
        run_async(endpoint, messages, args.temperature, args.max_tokens,
                   timeout=args.timeout)
    else:
        run_sync(endpoint, messages, args.temperature, args.max_tokens,
                 timeout=args.timeout)


if __name__ == "__main__":
    main()
