# Spec Delta

## Purpose

Provides the native macOS (arm64) MLX inference runner for the TLC extension's `mlx` local engine kind: a self-contained release binary published by this repository, communicating over stdin/stdout with newline-delimited JSON (the same protocol as the OpenVINO GenAI runner) and running mlx-community 4-bit models from local directories with streamed token output.

## ADDED Requirements

### Requirement: macOS arm64 release binary
The repository SHALL produce a self-contained macOS arm64 runner binary named `tlc-mlx-runner-macos-arm64` — locally via the build tasks (`build/` directory) and in CI; `v*` tags SHALL publish it as a GitHub release asset. The binary SHALL require no runtime library downloads.

#### Scenario: Local build output
- **WHEN** `task -t tasks.yaml build` completes on a macOS arm64 host
- **THEN** `build/tlc-mlx-runner-macos-arm64` exists and is executable

#### Scenario: Release publication
- **WHEN** a `v*` tag is pushed
- **THEN** CI builds the binary, smoke-tests it, and attaches `tlc-mlx-runner-macos-arm64` to a GitHub release

### Requirement: NDJSON stdio protocol
The runner SHALL read one request per stdin line as a JSON object (`generate` with `id`/`prompt`/`maxTokens`/`stop`, `cancel`, `shutdown`) and SHALL write newline-delimited JSON events to stdout: one `token` event per generated text piece, one `done` event at the end of a generation, and `error` events on failures. Human-readable logs SHALL go to stderr only. With missing or invalid arguments the runner SHALL emit one `error` event and exit with code 2.

#### Scenario: Streaming generation
- **WHEN** a `generate` request is written to the runner's stdin
- **THEN** zero or more `token` events followed by a `done` event arrive on stdout, in order

#### Scenario: Logs never corrupt the stream
- **WHEN** the runner logs diagnostics during generation
- **THEN** the log lines appear on stderr and stdout remains valid NDJSON

#### Scenario: Sequential requests
- **WHEN** a `generate` request arrives while a generation is in progress
- **THEN** the runner emits an `error` event (`busy`) and the running generation is unaffected

#### Scenario: Cancel
- **WHEN** a `cancel` request arrives mid-generation
- **THEN** generation stops and a `done` event with `finishReason:"cancel"` follows

#### Scenario: No arguments
- **WHEN** the runner binary is started without `--model`
- **THEN** it emits a `{"type":"error"}` event and exits with code 2

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
