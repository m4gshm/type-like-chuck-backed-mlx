# Type-Like-Chuck MLX runner

Native stdio runner used by the TLC VS Code extension as the `mlx` local
engine kind on **macOS arm64**. Loads an `mlx-community` 4-bit model directory
through [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) and serves
FIM completions over newline-delimited JSON on stdin/stdout. MLX is statically
linked and Metal is a system framework, so the binary is self-contained — no
runtime downloads.

## Protocol

Same contract as the OpenVINO GenAI runner (`type-like-chuck-backed-openvino`) —
the TLC extension speaks one protocol for both runners. Runner-specific
differences: MLX is always on `gpu` (no `--device`/`--prop` options), and the
`version` event carries no runtime field because the binary is self-contained.

Spawn: `tlc-mlx-runner --model <model-dir> [--ctx N]`. `tlc-mlx-runner --version`
prints a `{"type":"version","app":"..."}` event and exits 0 without loading a
model. `app` is the runner's own version from git: an exact tag on HEAD
(`v1.2.3`, `-dirty` for a modified tree), otherwise
`<tag>-<N>-g<sha>+<branch>.<build-time>`.

Events on stdout (one JSON object per line):
- `{"type":"version","app":"..."}` for `--version` (before any model load);
- `{"type":"ready","model":...,"device":"gpu","ctx":N}` after the model loads;
- `{"type":"token","text":"..."}` per generated piece (no id — only one
  generation is active);
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel"}`;
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...]}`,
`{"type":"cancel","id":N}` (cancels the active generation, or a queued one by id —
the cancelled queued request is answered immediately with
`{"type":"done","id":N,"finishReason":"cancel"}`), and `{"type":"shutdown"}` (drains
the queue answering every pending request with `finishReason:"cancel"`, cancels
the active generation and exits). Logs go to stderr only. The runner queues
generate requests and serves them one at a time in arrival order: `token`
events carry no id (only one generation is active) and always belong to the
request being processed; `done`/`error` events carry the request id. A failed
generation is reported with a single `error` event (no `done` follows). Exit
codes: 0 = normal, 1 = model load failure, 2 = usage/loader error.

## Build

Requires macOS arm64 (MLX does not cross-compile) and
[go-task](https://taskfile.dev):

```sh
task -t tasks.yaml build        # → build/tlc-mlx-runner-macos-arm64
task -t tasks.yaml smoke        # no-args JSON smoke
task -t tasks.yaml smoke:version  # --version smoke
MODEL_DIR=/path/to/model task -t tasks.yaml smoke:model   # NDJSON session
```

CI (`.github/workflows/runner.yml`) builds on `macos-latest`, runs the smoke
test, and publishes the binary as a GitHub release on `v*` tags. The TLC
plugin imports it from the latest release (or a pinned `RUNNER_VERSION` tag)
when packing its `.vsix`.
