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

## Benchmark

`task e2e:bench` measures model load time and inference speed (FIM prompt,
64 tokens, steady state of 2 runs) across the 4-bit/8-bit Qwen2.5-Coder line:

```text
== gpu ==  host=MacBook Pro  cpu=Apple M4 Pro  ram=48 GB  os=macOS 26.7.1  ctx=0  runs=2  maxTokens=64
model      load_s  ttft_s  gen_s  tokens  tok_s
0.5B-4bit     0.7    0.03   0.19     64   345.8
1.5B-4bit     0.4    0.05   0.39     64   165.8
3B-4bit       0.5    0.10   0.66     64    96.6
0.5B-8bit     0.3    0.02   0.24     64   263.5
1.5B-8bit     0.5    0.05   0.58     64   110.7
3B-8bit       0.9    0.10   1.09     64    58.7
7B-4bit       0.9    0.21   1.35     64    47.6
7B-8bit       1.6    0.21   2.31     64    27.7
```

`load_s` — time to the `ready` event, `ttft_s` — time to first token,
`tok_s` — generation throughput. Filter models with `SIZES=3B-4bit task e2e:bench`.

## OpenVINO runner benchmark

Same bench for the OpenVINO GenAI runner (`type-like-chuck-backed-openvino`),
for comparison. Intel laptop, NPU vs iGPU, Qwen2.5-Coder int8/int4/nf4 line:

```text
== NPU ==  host=ASUS EXPERTBOOK B9406CAA  cpu=Intel(R) Core(TM) Ultra X7 358H  ram=63 GB  os=Windows 11 Pro 26H2  name=Intel(R) AI Boost  ctx=0  runs=2  maxTokens=64  props=runner defaults
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-int8           1.4    0.12   1.80     43    23.9
1.5B-int8           3.0    0.41   6.85     49     7.2
3B-int8             5.2    1.16   4.18     11     2.6
0.5B-nf4            0.7    0.06   0.18     16    87.0
1.5B-nf4            1.7    0.10   0.94     51    54.3
3B-nf4              2.8    0.18   1.57     44    28.1
0.5B-int8-sym       1.3    0.04   0.57     51    89.4
1.5B-int8-sym       2.1    0.12   1.37     51    37.3
3B-int8-sym         4.4    0.16   1.55     27    17.4
0.5B-int4-sym       0.8    0.08   0.49     53   108.0
1.5B-int4-sym       1.5    0.16   1.06     51    48.1
3B-int4-sym         2.8    0.31   1.16     32    27.5

== GPU ==  host=ASUS EXPERTBOOK B9406CAA  cpu=Intel(R) Core(TM) Ultra X7 358H  ram=63 GB  os=Windows 11 Pro 26H2  name=Intel(R) Arc(TM) B390 GPU (iGPU)  ctx=0  runs=2  maxTokens=64  props=runner defaults
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-int8           0.7    0.04   0.49     43    87.4
1.5B-int8           1.1    0.06   1.15     49    42.8
3B-int8             2.1    0.10   0.41     11    26.9
0.5B-int4           0.7    0.02   0.37     43   116.6
1.5B-int4           0.9    0.04   0.29     20    69.7
3B-int4             1.5    0.08   0.27     11    41.3
0.5B-nf4            0.8    0.04   0.25     16    65.2
1.5B-nf4            1.8    0.10   1.92     56    29.1
3B-nf4              3.3    0.20   3.89     49    12.6
0.5B-int8-sym       0.7    0.02   0.43     51   118.8
1.5B-int8-sym       1.0    0.06   1.09     51    46.9
3B-int8-sym         1.7    0.04   1.18     27    22.8
0.5B-int4-sym       0.6    0.02   0.35     45   129.5
1.5B-int4-sym       0.8    0.04   0.68     51    75.5
3B-int4-sym         1.3    0.08   1.29     53    41.1
```

`compile_s` — time to the `ready` event (includes OpenVINO graph compilation;
NPU blobs are cached across runs), rest as above.
