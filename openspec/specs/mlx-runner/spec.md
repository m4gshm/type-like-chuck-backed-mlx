# mlx-runner Specification

## Purpose

Provides the native macOS (arm64) MLX inference runner for the TLC extension's `mlx` local engine kind: a self-contained release binary published by this repository, communicating over stdin/stdout with newline-delimited JSON (the same protocol as the OpenVINO GenAI runner) and running mlx-community 4-bit models from local directories with streamed token output.

## Requirements

### Requirement: macOS arm64 release binary
The repository SHALL produce a self-contained macOS arm64 runner binary named `tlc-mlx-runner-macos-arm64` — locally via the build tasks (`build/` directory) and in CI; `v*` tags SHALL publish it as a GitHub release asset. The binary SHALL require no runtime library downloads.

#### Scenario: Local build output
- **WHEN** `task -t tasks.yaml build` completes on a macOS arm64 host
- **THEN** `build/tlc-mlx-runner-macos-arm64` exists and is executable

#### Scenario: Release publication
- **WHEN** a `v*` tag is pushed
- **THEN** CI builds the binary, smoke-tests it, and attaches `tlc-mlx-runner-macos-arm64` to a GitHub release

### Requirement: NDJSON stdio protocol
The runner SHALL read one request per stdin line as a JSON object (`generate` with `id`/`prompt`/`maxTokens`/`stop`, an optional `cache` flag defaulting to false, `cancel`, `shutdown`) and SHALL write newline-delimited JSON events to stdout: a `version` event for `--version`, one `token` event (without id) per generated text piece, one `done` event at the end of a generation carrying `promptTokens` (the tokenized prompt length) and `cachedTokens` (prompt tokens served from the retained KV cache; 0 without reuse), and `error` events on failures. Human-readable logs SHALL go to stderr only. With missing or invalid arguments the runner SHALL emit one `error` event and exit with code 2.

#### Scenario: Version
- **WHEN** the runner is started with `--version`
- **THEN** a `{"type":"version"}` event is printed and the runner exits 0 without loading a model

#### Scenario: Streaming generation
- **WHEN** a `generate` request is written to the runner's stdin
- **THEN** zero or more `token` events followed by a `done` event arrive on stdout, in order

#### Scenario: Logs never corrupt the stream
- **WHEN** the runner logs diagnostics during generation
- **THEN** the log lines appear on stderr and stdout remains valid NDJSON

#### Scenario: Queued requests
- **WHEN** a `generate` request arrives while a generation is in progress
- **THEN** it is queued and served after the active generation, in arrival order

#### Scenario: Cancel active generation
- **WHEN** a `cancel` request arrives mid-generation
- **THEN** generation stops and a `done` event with `finishReason:"cancel"` follows

#### Scenario: Cancel queued request
- **WHEN** a `cancel` request arrives with the id of a queued request
- **THEN** that request is answered immediately with a `done` event with `finishReason:"cancel"` and is never generated

#### Scenario: Shutdown drains the queue
- **WHEN** a `shutdown` request arrives
- **THEN** every queued request is answered with a `done` event with `finishReason:"cancel"`, the active generation is cancelled, and the runner exits 0 without waiting longer than the cancelled generation's own teardown

#### Scenario: Failed generation
- **WHEN** generation fails with an error
- **THEN** the runner emits one `error` event with the request id and no `done` event follows

#### Scenario: No arguments
- **WHEN** the runner binary is started without `--model`
- **THEN** it emits a `{"type":"error"}` event and exits with code 2

#### Scenario: Prompt token counts reported
- **WHEN** a generation finishes with a `done` event
- **THEN** the event carries `promptTokens` (the tokenized prompt length) and `cachedTokens` (0 when no cache was reused)

#### Scenario: Prompt cache reuse
- **WHEN** a `generate` request with `"cache": true` shares at least 32 prompt tokens with the previous cleanly finished generation
- **THEN** the shared prefix is served from the retained KV cache instead of being re-prefilled, and the `done` event reports the reused count in `cachedTokens`

#### Scenario: Cache invalidated by an interrupted generation
- **WHEN** the previous generation ended by stop string, cancellation, or error
- **THEN** the retained cache is not reused for the next request and its `done` event reports `cachedTokens: 0`

### Requirement: Local-directory model inference contract
The runner SHALL load a model from a local directory (config.json, safetensors weights, tokenizer — `mlx-community` 4-bit format) given by `--model`, without any Hugging Face Hub access. Per request it SHALL accept a prompt string, a max-token limit and stop strings, and generate a single completion honoring the stop strings; a stop string SHALL never appear in the emitted text.

#### Scenario: Stop string ends generation
- **WHEN** the model generates text containing a request stop string
- **THEN** generation stops, the stop string is not included in the emitted tokens, and a `done` event with `finishReason:"stop"` follows

#### Scenario: Max tokens reached
- **WHEN** generation reaches the request `maxTokens` limit without a stop condition
- **THEN** a `done` event with `finishReason:"length"` follows

#### Scenario: Model directory missing
- **WHEN** the `--model` path does not contain a loadable model
- **THEN** the runner emits an `error` event and exits non-zero
