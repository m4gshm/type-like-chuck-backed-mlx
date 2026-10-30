# Type-Like-Chuck MLX runner

Native stdio runner used by the TLC VS Code extension as the `mlx` local
engine kind on **macOS arm64**. Loads an `mlx-community` 4-bit model directory
through [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) and serves
FIM completions over newline-delimited JSON on stdin/stdout. MLX is statically
linked and Metal is a system framework, so the binary is self-contained — no
runtime downloads.

## Protocol

Identical to the OpenVINO GenAI runner (`type-like-chuck-backed-openvino`).

Spawn: `tlc-mlx-runner --model <model-dir> [--ctx N]`

Events on stdout (one JSON object per line):
- `{"type":"ready","model":...,"device":"gpu","ctx":N}` after the model loads;
- `{"type":"token","text":"..."}` per generated piece;
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel"|"error"}`;
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...]}`,
`{"type":"cancel","id":N}`, `{"type":"shutdown"}`. Logs go to stderr only.
One generation at a time; a request arriving mid-generation is rejected with
an `error` event (`busy`). Started without `--model` the runner emits one
`error` event and exits with code 2.

## Build

Requires macOS arm64 (MLX does not cross-compile) and
[go-task](https://taskfile.dev):

```sh
task -t tasks.yaml build        # → build/tlc-mlx-runner-macos-arm64
task -t tasks.yaml smoke        # no-args JSON smoke
MODEL_DIR=/path/to/model task -t tasks.yaml smoke:model   # NDJSON session
```

CI (`.github/workflows/runner.yml`) builds on `macos-latest`, runs the smoke
test, and publishes the binary as a GitHub release on `v*` tags. The TLC
plugin imports it from the latest release (or a pinned `RUNNER_VERSION` tag)
when packing its `.vsix`.
