# llama_runpod — Runpod serverless image for llama.cpp

Docker image that runs a [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` instance behind a Runpod serverless worker.

The image is built by the
[`ghr-build-deploy-image` workflow](../.github/workflows/ghr-build-deploy-image.yaml)
and pushed to GHCR as `ghcr.io/<owner>/llama-runpod:latest` (where `<owner>` is
the GitHub user or org that owns the repository running the workflow).

---

## What's in the container

| File | Purpose |
| --- | --- |
| [`handler.py`](handler.py) | Runpod serverless worker. Forwards jobs to llama-server. |
| [`start_llama.sh`](start_llama.sh) | Boots `llama-server` → waits for `/health` → `exec`s the Runpod handler. |
| [`Dockerfile`](Dockerfile) | `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` + Python + `runpod>=1.12.0`. |

```
   ┌──────────────┐    SSE     ┌────────────────┐  HTTP/JSON+SSE  ┌──────────────┐
   │  Runpod /run │ ─────────▶ │  handler.py    │ ──────────────▶ │ llama-server │
   │  /runsync    │  yield     │  (generator)   │                 │  :8080       │
   │  /stream     │            └────────────────┘                 └──────────────┘
   └──────────────┘
```

The Runpod handler is the worker's job entry point. It forwards job input to
llama-server's OpenAI-compatible `/v1/chat/completions` endpoint and returns
the response as job output.

---

## Endpoints

### Runpod serverless worker (port 8000 inside the container, managed by Runpod)

| Operation | Behavior |
| --- | --- |
| `POST /run` with `"stream": true` | Returns a job id; stream chunks via `GET /stream/<id>` |
| `POST /run` without `stream` | Returns the full assembled response when complete |
| `POST /runsync` | Same as `/run` but blocks until completion |
| `GET /stream/<id>` | SSE stream of yielded chunks (`{"output": <chunk>}` per frame) |
| `GET /ping` | Health check (proxies llama-server's `/health`) |

---

## Streaming

## Streaming

Set `stream: true` in the job input. The handler is a Python generator that
yields llama-server SSE chunks one by one. Each yield is forwarded to the
caller's `GET /stream/<id>` as it is produced — no buffering.

````python
import runpod

runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("ENDPOINT_ID")

run_request = endpoint.run({
    "input": {
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
})

for chunk in run_request.stream():
    # chunk: {"output": {...llama-server chunk...}, "metrics": {...}}
    output = chunk.get("output", {})
    delta = output.get("choices", [{}])[0].get("delta", {})
    text = delta.get("content") or ""
    print(text, end="", flush=True)
````

Non-stream callers still work: `POST /run` without `"stream": true` returns
the full JSON response at completion.

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
| `LLAMA_TIMEOUT` | no | `300` | Per-request timeout for the handler's non-stream POST |
| `LLAMA_STREAM_TIMEOUT` | no | _(unset)_ | Read timeout for the handler's streaming POST. **Unset = no read timeout.** |
| `HF_HOME` | no | `/runpod-volume/huggingface-cache` | Hugging Face cache directory (mount a network volume here to keep models across cold starts) |

### Set on the GitHub Actions workflow (build pipeline)

See the top-level [README.md](../README.md#runpod-configuration-llama-build)
for `RUNPOD_*` workflow variables (pod sizing, runner image, etc.).

---

## Cold start / model load

`start_llama.sh` performs two sequential waits before the Runpod handler
is ready to accept jobs:

1. **`llama-server` start** — up to **600s** (10 min). Polls `GET /health`
   every 5s. Fails fast (dumps the log, `exit 1`) if the process dies.
2. **`exec python3 -u /app/handler.py`** — the Runpod worker takes over.

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

The handler forwards only the keys llama-server understands
(see `_FORWARD_FIELDS` in [handler.py](handler.py)). Unknown keys are
silently dropped. `messages` is required.

Defaults (`temperature=0.7`, `max_tokens=256`) are only applied on the
**non-stream** path; for streaming, omit fields you don't want to set
because llama-server treats an omitted `max_tokens` as "no limit".

---

## Local development

The handler has no llama-server dependency at import time, so you can
smoke-test it with monkey-patched HTTP clients.

```sh
# Throwaway venv with the production deps
uv venv /tmp/runpod-smoke --python 3.11
VIRTUAL_ENV=/tmp/runpod-smoke uv pip install \
    'runpod>=1.12.0' requests httpx
```

Notes for development:

- `runpod.serverless.start({...})` runs at import time. Patch it to a
  no-op before importing the handler, otherwise it tries to launch a
  worker and exits with `WARN | test_input.json not found, exiting.`
- `runpod`'s import chain pulls in `requests.Session`, `RequestException`,
  and `ConnectionError`, so a stub `requests` module is not enough —
  install the real package and monkey-patch only `post` / `get`.

---

## Logs and debugging

| Log file | Source |
| --- | --- |
| `/tmp/llama-server.log` | llama-server stdout/stderr |
| stdout | Runpod worker (`python3 -u /app/handler.py`) |

To peek at a running container:

```sh
runpodctl exec <pod-id> -- tail -f /tmp/llama-server.log
```

---

## See also

- [Top-level README](../README.md) — build workflows, GitHub App setup,
  pod resource configuration.
- [llama.cpp docs](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)
  — full set of `llama-server` flags and request body fields.
- [Runpod serverless docs](https://docs.runpod.io/serverless) —
  endpoint creation, scaling, and pricing.
