# Proposal

## Why

The TLC extension's `ovgenai` engine kind explicitly does not target macOS. macOS (Apple Silicon) is the primary dev platform for the plugin, so local FIM completions there need a native engine. MLX is Apple's array framework for Apple silicon; `mlx-community` publishes ready 4-bit quantized LLMs (including Qwen2.5-Coder) in a directory format loadable without conversion.

## What Changes

- Add a **native MLX runner** for macOS arm64: a Swift executable speaking the **same NDJSON stdin/stdout protocol** as the OpenVINO GenAI runner — completion requests in on stdin, streamed token events out on stdout, logs out on stderr. No ports, no HTTP, no API keys.
- Loads models from a **local directory** (`config.json`, safetensors weights, tokenizer) — `mlx-community` 4-bit format, no GGUF/IR conversion.
- Sequential request handling (one generation at a time), mid-generation requests rejected with a `busy` error event; `cancel` stops the current generation; stop strings end generation and are excluded from the emitted text.
- MLX is statically linked and Metal is a system framework — **no runtime library downloads**; the binary is self-contained.
- Build automation (`tasks.yaml`, macOS arm64 only) and CI (`macos-latest`) with a no-args JSON smoke test; `v*` tags publish `tlc-mlx-runner-macos-arm64` as a GitHub release asset, from which the TLC plugin imports it.

## Capabilities

### New Capabilities

- `mlx-runner`: the native macOS stdio runner — NDJSON protocol over stdin/stdout, release binary, local-directory model inference contract (streaming tokens, stop strings, cancellation).

## Impact

- New Swift component: NDJSON stdio server on MLX Swift LM (`MLXLMCommon.generate`).
- Downstream: the TLC plugin imports the binary from this repo's `build/` directory (local) or GitHub release (CI) into `binaries/mlx/darwin-arm64/`; the plugin's stdio transport (`stdio_runner.ts`) is reused unchanged — only a third `engineKind: "mlx"` value and a darwin-arm64 platform guard are added there.
