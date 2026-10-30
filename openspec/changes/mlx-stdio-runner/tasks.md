# Tasks

## 1. Scaffold

- [x] 1.1 Swift Package skeleton: `Package.swift` (macOS 14+, mlx-swift-lm dependency), executable target, `.gitignore` (build/, .cache/, .build/), README with protocol and build instructions (artifact: package resolves; verified by `swift package resolve`)

## 2. Runner implementation

- [ ] 2.1 NDJSON wire layer: line reader on stdin, JSON event writer on stdout/stderr, arg parsing (`--model`, `--ctx`), no-args behavior (exit 2 + one `error` event) (artifact: `main.swift`; verified by `echo | swift run` smoke)
- [ ] 2.2 Model load + ready handshake from a local model directory via `LLMModelFactory`/`ModelConfiguration(directory:)`; `ready` event with model name, device `gpu`, ctx (artifact: code; verified against a real mlx-community 4-bit model)
- [ ] 2.3 Generation loop: `generate` request handling (prompt, maxTokens, stop strings), streamer emitting `token` events with hold-back buffering so stop strings never leak, `done` with `finishReason` stop/length (artifact: code; verified by streaming a fixed FIM prompt)
- [ ] 2.4 `cancel` (Task cancellation → `finishReason:"cancel"`), sequential policy (`busy` rejection), `shutdown`/EOF exit 0 (artifact: code; verified by scripted stdin session)

## 3. Build automation and CI

- [ ] 3.1 `tasks.yaml`: build → `build/tlc-mlx-runner-macos-arm64`, smoke, clean (artifact: `tasks.yaml`; verified by a local build + no-args smoke)
- [ ] 3.2 CI workflow on `macos-latest`: build, no-args smoke, artifact upload; `v*` tags publish a GitHub release with `tlc-mlx-runner-macos-arm64` (artifact: `.github/workflows/runner.yml`; verified by the first tag run producing an importable release)
