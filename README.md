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
task build            # → build/tlc-mlx-runner-macos-arm64
task smoke            # no-args JSON smoke
task smoke:version    # --version smoke
MODEL_DIR=/path/to/model task smoke:model   # NDJSON session
```

CI (`.github/workflows/runner.yml`) builds on `macos-latest`, runs the smoke
test, and publishes the binary as a GitHub release on `v*` tags. The TLC
plugin imports it from the latest release (or a pinned `RUNNER_VERSION` tag)
when packing its `.vsix`.

## Real file experiment

`task real:file` runs raw FIM against the bundled fixture
(`scripts/fixtures/api_test.go`, cursor at line 96, column 9) with the plugin's
prompt shape (qwen2 markers, 8000/2000 clamp, reserved-token masking) but no
newline stop — it shows the backend's natural, possibly multi-line completion
plus an overlap report against the line after the cursor. `FILE`, `LINE`,
`COLUMN`, `MAX_TOKENS`, `MODEL_DIR` override the defaults.

## Benchmark

`task e2e:bench` measures model load time and inference speed across the
4-bit/8-bit Qwen2.5-Coder line. The default prompt is a plugin-shaped FIM over
the bundled fixture (4815 chars ≈ 1.5–2k token prefill — what the plugin
actually sends); `--prompt tiny` switches to the one-line FIM.
64 generated tokens, steady state of 2 runs:

```text
== gpu ==  host=MacBook Pro  cpu=Apple M4 Pro  ram=48 GB  os=macOS 26.7.1  ctx=0  runs=2  maxTokens=64  prompt=4815 chars
model      load_s  ttft_s  gen_s  tokens  tok_s
0.5B-4bit     0.4    0.32   0.53     52    98.0
1.5B-4bit     0.4    0.95   1.33     60    45.1
3B-4bit       0.5    2.00   2.65     62    23.4
0.5B-8bit     0.5    0.31   0.58     53    91.3
1.5B-8bit     0.5    0.98   1.57     49    31.3
3B-8bit       0.9    2.02   3.07     59    19.2
7B-4bit       1.2    4.75   6.04     62    10.3
7B-8bit       2.4    5.06   7.31     62     8.5

(one-line prompt, same host:)
== gpu ==  host=MacBook Pro  cpu=Apple M4 Pro  ram=48 GB  os=macOS 26.7.1  ctx=0  runs=2  maxTokens=64  prompt=78 chars
model      load_s  ttft_s  gen_s  tokens  tok_s
0.5B-4bit     0.4    0.02   0.09     30   321.4
1.5B-4bit     0.4    0.05   0.36     64   179.3
3B-4bit       0.5    0.07   0.63     64   101.3
0.5B-8bit     0.5    0.03   0.24     64   261.3
1.5B-8bit     0.5    0.05   0.12     11    88.4
3B-8bit       0.9    0.07   1.05     64    60.8
7B-4bit       1.1    0.12   0.74     32    43.3
7B-8bit       2.3    0.12   2.25     64    28.4
```

`load_s` — time to the `ready` event, `ttft_s` — time to first token,
`tok_s` — generation throughput. Filter models with `SIZES=3B-4bit task e2e:bench`.
