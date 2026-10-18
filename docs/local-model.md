# Real-model CPU integration

This milestone validates the gateway's backend contract with one real CPU replica.
Two simulated replicas exercise routing and failure behavior. Neither experiment
establishes vLLM GPU cache residency or GPU performance.

## Prepare and run the compatibility check

From the repository root on macOS arm64:

```bash
uv sync --locked --extra dev --extra tokenizer
uv run python scripts/prepare_llama.py
uv run --extra tokenizer python scripts/smoke_llama.py
```

The first preparation downloads about 2.6 GB into `.local/`, verifies SHA256 checks,
and writes `.local/manifest.json`. Reruns verify existing files rather than downloading
them again. The tokenizer is loaded locally, with remote Python code disabled, and
does not load PyTorch or model weights into the gateway.

| Artifact | Pinned identity |
|---|---|
| llama.cpp official Mac ARM runtime | `b11719`, build commit `5aeac37a3` |
| Runtime archive SHA256 | `076a2b0d15253078b90bd520a82dc2edc6c17657968878876ba3def3e80f54ba` |
| Community GGUF repository | `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` |
| GGUF repository revision | `4edb920b6f14e3b9284d4502a6485103d72cde05` |
| Model file | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| Model SHA256 | `8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc` |
| Original tokenizer repository | `Qwen/Qwen3-4B-Instruct-2507` |
| Tokenizer revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |

The runtime is published by llama.cpp. The GGUF is a community quantization of the
original Qwen model; its provenance is distinct from the original tokenizer. The
preparer pins and verifies all three tokenizer/config files separately. Large model
files, runtime binaries, service logs and the manifest remain ignored in Git.

## What the check verifies

1. llama.cpp starts with CPU-only flags, a 2,048-token context and one execution slot.
2. The gateway loads the prepared tokenizer and constructs a namespace from the model
   SHA256, tokenizer serialization fingerprint and chat-template fingerprint.
3. For a long system prompt and a multilingual, multi-turn conversation, the script
   compares the full Hugging Face token-ID sequence with llama.cpp's
   `/apply-template` followed by `/tokenize`. A mismatch fails the check.
4. A normal completion establishes an affinity hint; a later streaming request
   follows that hint, has actual backend usage, and completes with `[DONE]`.
5. A live generation is cancelled after its first content event. The script waits
   for both the gateway reservation and llama.cpp's `/slots` execution state to
   become idle, then checks that another request succeeds.
6. The script stops both child processes, even when a check fails.

Evidence is saved in `results/examples/real-cpu-smoke.json`, including artifact
identity, CPU flags, tokenizer fingerprints, actual output, usage and cleanup checks.
Runtime logs are `.local/llama-smoke-server.log` and `.local/llama-smoke-gateway.log`.
The script uses temporary local ports to avoid interfering with the simulator.

Matching token IDs is verified for the tested request subset and pinned pair. It
does not prove that every future template/backend combination matches. Repeat this
check when changing tokenizer files, chat templates, model artifacts or engine
versions. GPU integration will need its own vLLM tokenization and metric checks.

## Keep the real-model service running

Run the worker in one terminal:

```bash
.local/llama/llama-b11719/llama-server \
  --model .local/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --alias qwen-local --host 127.0.0.1 --port 8201 \
  --ctx-size 2048 --parallel 1 --threads 4 \
  --device none --n-gpu-layers 0 --no-op-offload --jinja --slots --metrics
```

Run the gateway in another:

```bash
uv run --extra tokenizer nano-inference gateway \
  --port 8000 --worker-urls http://127.0.0.1:8201 \
  --model qwen-local --policy prefix_affinity --capacity 1 --request-timeout 60 \
  --prefix-tokenizer .local/tokenizer --prefix-tokens 64 \
  --model-revision sha256:8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc
```

Send a request:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-local","messages":[{"role":"user","content":"Name two colors."}],"stream":true,"max_tokens":16,"stream_options":{"include_usage":true}}'
```

Short requests will report `prefix_too_short` with a 64-token prefix; use a longer
shared system prompt or the repeated-prefix workload to exercise affinity. Keep
prompt plus output within this CPU worker's context. Stop each service with Ctrl-C.

References: [llama.cpp server API](https://github.com/ggml-org/llama.cpp/tree/b11719/tools/server),
[GGUF artifact](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/tree/4edb920b6f14e3b9284d4502a6485103d72cde05),
[original tokenizer](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554).
