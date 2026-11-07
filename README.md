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
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel","promptTokens":N,"cachedTokens":N}`
  (prompt token count from the runner's tokenizer — omitted on the cancel paths
  that answer without generating — and how many of them came from the KV cache,
  0 when the request ran without `cache` or missed);
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...],"cache":true|false}`,
`{"type":"cancel","id":N}` (cancels the active generation, or a queued one by id —
the cancelled queued request is answered immediately with
`{"type":"done","id":N,"finishReason":"cancel"}`), and `{"type":"shutdown"}` (drains
the queue answering every pending request with `finishReason:"cancel"`, cancels
the active generation and exits). `cache` opts the request into KV-cache reuse:
when its prompt shares a long token prefix with the previous cached generation
(≥32 tokens — typing at the cursor grows the FIM prefix), the shared rows are
trimmed to the common prefix and only the rest is prefilled; the `done` event
reports how many prompt tokens the cache served. The cache holds one entry
(the last clean generation) and is rebuilt after a cancelled or stop-string
generation, a failed one, or a prompt that no longer matches. Logs go to stderr only. The runner queues
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
the bundled fixture (4815 chars = 2773 prefix + 2000 suffix ≈ 1.7k tokens —
what the plugin actually sends); `--prompt tiny` switches to the one-line FIM.
64 generated tokens, steady state of 2 runs. `ttft_s` carries the prefill;
`gen_s`/`tok_s` are decode-only (end-to-end first suggestion ≈ ttft_s + gen_s):

```text
== gpu ==  host=MacBook Pro  cpu=Apple M4 Pro  ram=48 GB  os=macOS 26.7.1  ctx=0  runs=2  maxTokens=64  prompt=4815 chars = 2773 prefix + 2000 suffix, 1724 tokens
model      load_s  ttft_s  gen_s  tokens  tok_s
0.5B-4bit     0.4    0.33   0.20     64   328.2
1.5B-4bit     0.4    1.06   0.41     61   149.6
3B-4bit       0.6    2.11   0.67     62    91.9
0.5B-8bit     0.4    0.47   0.27     60   223.0
1.5B-8bit     0.5    1.03   0.60     57    95.7
3B-8bit       1.0    2.30   1.10     59    53.4
7B-4bit       1.2    4.79   1.27     62    48.7
7B-8bit       2.3    5.35   2.38     62    26.1

warm start (KV-cache reuse, prompt + typed characters):
model      ttft_w_s  cached_tok
0.5B-4bit      0.16  891/1726
1.5B-4bit      0.53  891/1726
3B-4bit        1.08  891/1726
0.5B-8bit      0.17  891/1726
1.5B-8bit      0.54  891/1726
3B-8bit        1.10  891/1726
7B-4bit        2.50  891/1726
7B-8bit        2.60  891/1726

(one-line prompt, 78 chars / 16 tokens:)
== gpu ==  host=MacBook Pro  cpu=Apple M4 Pro  ram=48 GB  os=macOS 26.7.1  ctx=0  runs=2  maxTokens=64  prompt=78 chars = 78 prefix + 0 suffix, 16 tokens
model      load_s  ttft_s  gen_s  tokens  tok_s
0.5B-4bit     0.5    0.02   0.18     64   350.7
1.5B-4bit     0.4    0.05   0.34     64   188.1
3B-4bit       0.4    0.06   0.60     64   105.8
0.5B-8bit     0.4    0.02   0.24     64   266.8
1.5B-8bit     0.5    0.05   0.18     23   126.0
3B-8bit       1.4    0.06   1.02     64    62.6
7B-4bit       1.3    0.12   0.90     48    53.3
7B-8bit       2.3    0.13   2.21     64    29.0
```

`load_s` — time to the `ready` event, `ttft_s` — time to first token
(prompt token count comes from the runner's `done` event), `tok_s` —
decode throughput. Filter models with `SIZES=3B-4bit task e2e:bench`.
With `--cache` every model also runs two warm continuations — the prompt
grows by typed characters the way typing does — and reports `ttft_w_s`
(first token of the warm run) and `cached_tok` (prompt tokens served from
the KV cache): the shared FIM prefix (~half the prompt here) is no longer
prefilled, so the warm first token lands about twice as fast.
