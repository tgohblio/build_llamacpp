# llama_runpod — Runpod serverless image for llama.cpp

Docker image that runs a [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` instance behind a Runpod serverless worker, with a small
OpenAI-compatible HTTP sidecar so OpenAI SDK clients can connect directly.

The image is built by the
[`ghr-build-deploy-image` workflow](../.github/workflows/ghr-build-deploy-image.yaml)
and pushed to GHCR as `ghcr.io/tgohblio/llama-runpod:latest`.

---

## What's in the container

| File | Purpose |
| --- | --- |
| [`handler.py`](handler.py) | Runpod serverless worker. Forwards jobs to llama-server. |
| [`oai_proxy.py`](oai_proxy.py) | FastAPI/uvicorn sidecar exposing `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`. Streams SSE. |
| [`start_llama.sh`](start_llama.sh) | Boots `llama-server` → waits for `/health` → boots the OAI proxy → `exec`s the Runpod handler. |
| [`Dockerfile`](Dockerfile) | `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` + Python + `runpod>=1.12.0` + `fastapi` + `uvicorn[standard]`. |

```
   ┌──────────────┐    SSE     ┌────────────────┐  HTTP/JSON+SSE  ┌──────────────┐
   │  Runpod /run │ ─────────▶ │  handler.py    │ ──────────────▶ │ llama-server │
   │  /runsync    │  yield     │  (generator)   │                 │  :8080       │
   │  /stream     │            └────────────────┘                 └──────────────┘
   └──────────────┘                       ▲
                                         │  HTTP/JSON+SSE
   ┌──────────────┐                       │
   │  OpenAI SDK  │ ──────────▶  ┌────────────────┐
   │  curl, etc.  │  /v1/...     │  oai_proxy.py  │ ──────────────▶ llama-server
   └──────────────┘              │  :8081         │
                                 └────────────────┘
```

Both clients can run side-by-side in the same container. The Runpod handler
is the worker's job entry point; the OAI proxy is a regular FastAPI app
serving the standard OpenAI API surface.

---

## Endpoints

### Runpod serverless worker (port 8000 inside the container, managed by Runpod)

Worker entry point: `runpod.serverless.start({"handler": handler, "ping": _ping_handler, "return_aggregate_stream": True})`.

| Operation | Behavior |
| --- | --- |
| `POST /run` with `"stream": true` | Returns a job id; stream chunks via `GET /stream/<id>` |
| `POST /run` without `stream` | Returns the full assembled response when complete |
| `POST /runsync` | Same as `/run` but blocks until completion |
| `GET /stream/<id>` | SSE stream of yielded chunks (`{"output": <chunk>}` per frame) |
| `GET /ping` | Health check (proxies llama-server's `/health`) |

### OAI-compatible sidecar (port `OAI_PORT`, default `8081`)

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Plaintext `"ok"` — used by `start_llama.sh` boot wait |
| `GET /v1/models` | Lists the configured `MODEL` (stubbed) |
| `POST /v1/chat/completions` | OpenAI-compatible chat completion (streaming + non-streaming) |
| `POST /v1/completions` | OpenAI-compatible legacy text completion (streaming + non-streaming) |

---

## Streaming

There are two streaming paths. Pick the one your client supports.

### 1. Runpod generator-based streaming (native Runpod)

Set `stream: true` in the job input. The handler is a Python generator that
yields llama-server SSE chunks one by one. Each yield is flushed to the
caller's `GET /stream/<id>` as it is produced — no buffering.

````python
import runpod

endpoint = runpod.Endpoint("ENDPOINT_ID")
req = endpoint.run({
    "input": {
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
})

for chunk in req.stream():
    # chunk is `{"output": <llama-server chunk dict>}`
    delta = chunk["output"]["choices"][0].get("delta", {})
    text = delta.get("content") or ""
    print(text, end="", flush=True)
````

Non-stream callers still work: `POST /run` without `"stream": true` returns
the full JSON response at completion (`return_aggregate_stream: True`).

The yielded chunks preserve OpenAI's delta shape exactly, including
`reasoning_content` for thinking models (Qwen3, QwQ, QVQ, Qwen-Omni):

| Field | Stage |
| --- | --- |
| `reasoning_content` set, `content` null | Thinking |
| `reasoning_content` null, `content` set | Responding |
| both null | Same stage as the previous chunk |

A final `{"done": True}` chunk is yielded in a `finally` block so the
consumer has a deterministic end-of-stream signal without parsing
`data: [DONE]`.

Mid-stream errors yield a final `{"error": "...", "choices": []}` chunk
before the `done` sentinel, so the caller sees the failure on `/stream`
instead of an opaque connection close.

### 2. OpenAI SDK streaming (via the OAI proxy)

Point the OpenAI Python (or Node, curl, anything) at the sidecar's
`/v1/chat/completions`:

````python
from openai import OpenAI

client = OpenAI(
    base_url="http://<endpoint-host>:8081/v1",
    api_key="not-used",  # proxy doesn't check; set MODEL via env on the endpoint
)

completion = client.chat.completions.create(
    model="any",  # proxy ignores; llama-server uses $MODEL from the endpoint env
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in completion:
    if chunk.choices:
        text = chunk.choices[0].delta.content or ""
        print(text, end="", flush=True)
    elif chunk.usage:
        print("\n--- usage ---", chunk.usage)
````

SSE framing from the sidecar is standards-compliant:

```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no        # disables nginx output buffering
Connection: keep-alive

data: {"choices":[{"delta":{"content":"Hel"},"index":0}]}

data: {"choices":[{"delta":{"content":"lo"},"index":0}]}

data: [DONE]
```

> If you put nginx in front of the sidecar, set
> `proxy_buffering off;` on the location block — otherwise nginx will buffer
> the chunks and break real-time streaming.

---

## Environment variables

### Set on the Runpod endpoint (consumed by the container)

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MODEL` | **yes** | — | HF model id, e.g. `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` |
| `DRAFT_MODEL` | **yes** | — | HF draft model id for speculative decoding |
| `N_GPU_LAYERS` | no | `99` | GPU layers to offload (`-ngl`) |
| `CTX_SIZE` | no | `8192` | Context size (`-c`) |
| `PARALLEL` | no | `1` | Number of parallel sequences (`--parallel`) |
| `PORT` | no | `8080` | llama-server listen port |
| `SPEC_TYPE` | no | _(unset)_ | `draft-dflash` to enable DFlash 2 speculative decoding |
| `SPEC_DRAFT_N_MAX` | no | `7` | Max draft tokens per step |
| `OAI_ENABLED` | no | `1` | Set `0` to skip the OAI sidecar (saves memory) |
| `OAI_PORT` | no | `8081` | Sidecar listen port |
| `OAI_LOG_LEVEL` | no | `info` | uvicorn log level (`critical`, `error`, `warning`, `info`, `debug`, `trace`) |
| `OAI_TIMEOUT` | no | `10` | Connect timeout (seconds) for proxy → llama-server hops |
| `OAI_READ_TIMEOUT` | no | _(unset)_ | Read timeout in seconds. **Unset = no read timeout.** Set only if you have a hard cap on stream duration. |
| `LLAMA_TIMEOUT` | no | `300` | Per-request timeout for the handler's non-stream POST |
| `LLAMA_STREAM_TIMEOUT` | no | _(unset)_ | Read timeout for the handler's streaming POST. **Unset = no read timeout.** |
| `HF_HOME` | no | `/runpod-volume/huggingface-cache` | Hugging Face cache directory (mount a network volume here to keep models across cold starts) |

### Set on the GitHub Actions workflow (build pipeline)

See the top-level [README.md](../README.md#runpod-configuration-llama-build)
for `RUNPOD_*` workflow variables (pod sizing, runner image, etc.).

---

## Cold start / model load

`start_llama.sh` performs three sequential waits before the Runpod handler
is ready to accept jobs:

1. **`llama-server` start** — up to **600s** (10 min). Polls `GET /health`
   every 5s. Fails fast (dumps the log, `exit 1`) if the process dies.
2. **OAI proxy start** — up to **30s**. Polls `GET /health` every 1s.
   Skipped entirely if `OAI_ENABLED=0`.
3. **`exec python3 -u /app/handler.py`** — the Runpod worker takes over.

To speed up cold starts, mount a Runpod network volume at
`/runpod-volume` and pre-warm it with the model files. The image sets
`HF_HOME` to `/runpod-volume/huggingface-cache` so HF will reuse the cache
on subsequent boots.

---

## Job input shape

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "stop": ["..."],
  "stream": false,
  "stream_options": {"include_usage": true},
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

Or, for legacy text completion:

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 64
}
```

The handler forwards only the keys llama-server understands
(see `_FORWARD_FIELDS` in [handler.py](handler.py)). Unknown keys are
silently dropped. At least one of `messages` or `prompt` is required.

Defaults (`temperature=0.7`, `max_tokens=256`) are only applied on the
**non-stream** path; for streaming, omit fields you don't want to set
because llama-server treats an omitted `max_tokens` as "no limit".

---

## Local development

The handler and OAI proxy have no llama-server dependency at import time,
so you can smoke-test both with monkey-patched HTTP clients.

```sh
# Throwaway venv with the production deps
uv venv /tmp/runpod-smoke --python 3.11
VIRTUAL_ENV=/tmp/runpod-smoke uv pip install \
    'runpod>=1.12.0' 'fastapi>=0.115' 'uvicorn[standard]>=0.30' requests httpx

# Drive the OAI proxy with FastAPI's TestClient (no live llama-server)
python -c "from fastapi.testclient import TestClient; ..."
```

Notes that bit during development:

- `runpod.serverless.start({...})` runs at import time. Patch it to a
  no-op before importing the handler, otherwise it tries to launch a
  worker and exits with `WARN | test_input.json not found, exiting.`
- `runpod`'s import chain pulls in `requests.Session`, `RequestException`,
  and `ConnectionError`, so a stub `requests` module is not enough —
  install the real package and monkey-patch only `post` / `get`.
- The OAI proxy's streaming branch uses `client.send(...)` and the
  non-streaming branch uses `client.post(...)`. A test fake must support
  both, plus `build_request(...)` and `aiter_lines()`.
- `httpx.Response.json()` is **synchronous** in httpx, even inside an
  async context. Don't `await upstream.json()`.

A full smoke test that exercises every endpoint and the error path
lives in the repo memory file
`/memories/repo/build_llamacpp.md` (search for "Local smoke testing").

---

## Logs and debugging

| Log file | Source |
| --- | --- |
| `/tmp/llama-server.log` | llama-server stdout/stderr |
| `/tmp/oai-proxy.log` | uvicorn (OAI sidecar) stdout/stderr |
| stdout | Runpod worker (`python3 -u /app/handler.py`) |

To peek at a running container:

```sh
runpodctl exec <pod-id> -- tail -f /tmp/llama-server.log
runpodctl exec <pod-id> -- tail -f /tmp/oai-proxy.log
```

---

## See also

- [Top-level README](../README.md) — build workflows, GitHub App setup,
  pod resource configuration.
- [llama.cpp docs](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)
  — full set of `llama-server` flags and request body fields.
- [Runpod serverless docs](https://docs.runpod.io/serverless) —
  endpoint creation, scaling, and pricing.
