# Design

## Context

The `ovgenai` runner targets Windows x64 + Linux x64 and explicitly excludes macOS. macOS dev hosts need local FIM completions too. Options evaluated: llama.cpp Metal builds (already the extension default elsewhere — GGUF conversion), `openvino-node` (no macOS release target), MLX (Apple's framework, first-class on Apple Silicon, `mlx-community` ships 4-bit Qwen2.5-Coder ready to run).

Decision: a Swift stdio runner on MLX Swift LM, speaking the ovgenai NDJSON protocol so the plugin's stdio transport is reused unchanged.

## Goals / Non-Goals

- Goals: FIM completion on macOS arm64 with streamed tokens, stop strings, cancellation; models from local directories (`mlx-community` 4-bit); self-contained binary with zero runtime downloads; identical wire protocol to the ovgenai runner.
- Non-Goals: Windows/Linux (ovgenai and llama.cpp own those); chat-template handling in the runner (the extension sends a fully rendered FIM prompt); vision/multimodal; fine-tuning.

## Decisions

### D1. Same NDJSON protocol as the ovgenai runner
The runner is a private child process; its stdin/stdout pipes are the transport. Protocol is byte-compatible with `type-like-chuck-backed-openvino` design D1:

- Spawn: `tlc-mlx-runner --model <model-dir> [--ctx N]` (no `--device` — Metal is the only backend).
- Requests in (stdin): `{"type":"generate","id":1,"prompt":"...","maxTokens":128,"stop":["\n"]}`, `{"type":"cancel","id":1}`, `{"type":"shutdown"}`.
- Events out (stdout): `{"type":"ready","model":"...","device":"gpu","ctx":N}`, `{"type":"token","id":1,"text":"..."}`, `{"type":"done","id":1,"finishReason":"stop"|"length"|"cancel"}`, `{"type":"error","id":1|null,"message":"..."}`.
- Logs (stderr): free-form lines, forwarded by the extension to the TLC output channel.
- With no/invalid args: exit code 2 after emitting one `error` event (the plugin's smoke test asserts this).

### D2. Swift + MLX Swift LM
C++ MLX has no LLM-pipeline layer; Python would drag a whole interpreter. `ml-explore/mlx-swift-lm` provides everything: `LLMModelFactory.shared.loadContainer(configuration: ModelConfiguration(directory: URL(filePath: modelDir)))` loads a local directory (config.json + safetensors + tokenizer) with no Hub access; `MLXLMCommon.generate(input:parameters:context:)` streams through a callback; `NaiveStreamingDetokenizer` turns token ids into text pieces; Swift structured concurrency gives cancellation via `Task.cancel()`.

### D3. Generation loop
- Per request: `UserInput(prompt:)` → `processor.prepare` → `generate` with `GenerateParameters(maxTokens:)`.
- Callback emits `token` events from the detokenizer; when the detokenized text contains a stop string, generation returns `.stop`.
- **Stop-string hygiene**: text is emitted through a hold-back buffer — the trailing `maxStopLength - 1` characters are withheld until disambiguated; on a stop hit, everything from the stop-string start onward is discarded, so the stop string never reaches the output.
- **finishReason**: cancellation (`Task.isCancelled` / `CancellationError`) → `"cancel"`; stopped at `maxTokens` → `"length"`; otherwise (EOS or stop string) → `"stop"`.
- **Cancel**: the generation runs in a stored `Task`; a `cancel` request cancels it; a `shutdown` request (or stdin EOF) exits 0 after the current task settles.
- **Sequential only**: a `generate` arriving mid-generation is rejected with an `error` event (`busy`), mirroring ovgenai; the extension serializes requests itself.

### D4. Packaging and CI
- Swift Package executable; `swift build -c release` produces a self-contained arm64 binary (MLX linked statically, Metal is a system framework) — no runtime archives, no library-path handling, unlike ovgenai.
- Build host must be macOS arm64 (MLX does not cross-compile; CI `macos-latest` is arm64-native).
- `tasks.yaml`: `build` → `build/tlc-mlx-runner-macos-arm64` (release-asset name, so plugin import needs no renaming), `smoke` (no-args JSON check; optional NDJSON session with `MODEL_DIR`), `clean`.
- CI `runner.yml`: build + no-args smoke + artifact; `v*` tags publish the binary via `softprops/action-gh-release`. The plugin imports it (its own CI runs on ubuntu and can only fetch, not build).

## Risks / Trade-offs

- [Swift 6 strict concurrency around a stateful worker] → the runner state lives in a single `actor`; the generate callback only touches nonisolated emit helpers.
- [Detokenizer granularity] → `token` events arrive per detokenizer piece, not per token id; the extension assembles incrementally either way (same as ovgenai).
- [mlx-swift-lm is a young package] → pinned via `from:` version in Package.swift; breaking API changes are fixed in this repo and shipped by a new release tag.
- [Model catalog duplication] → the plugin owns model discovery/download; this repo never talks to Hugging Face.
