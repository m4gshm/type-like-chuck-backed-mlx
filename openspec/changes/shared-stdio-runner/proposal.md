# Proposal

## Why

Each VS Code window spawns its own runner process and loads its own copy of the model, so two windows double the (unified) memory footprint for the same model on the same Mac. The runner today speaks NDJSON over private stdin/stdout pipes, which no second client can attach to.

## What Changes

- **Listen mode**: the runner gains `--listen <endpoint>` — it serves the same NDJSON protocol over a unix domain socket instead of stdin/stdout, without opening any TCP port.
- **Client refcount and self-exit**: the runner tracks connected clients and exits by itself when the last client disconnects (after having had at least one); the spawning window closing does not terminate the runner while other clients remain.
- **Per-request token routing**: `token` events carry the request `id` so two concurrent clients cannot receive each other's tokens.
- **PID exposure**: the runner reports its PID on the `version`/ready handshake so clients can validate backend liveness and display it.
- Parity with the OpenVINO runner: the identical listen-mode contract is specified in `type-like-chuck-backed-openvino` (same change name `shared-stdio-runner`), so the TLC extension speaks one shared-runner protocol with both backends.

## Capabilities

### New Capabilities

- `runner-sharing`: the shared/listen-mode lifecycle — listen endpoint, client refcount with self-termination, per-client stream isolation, PID exposure.

### Modified Capabilities

- `mlx-runner`: the "NDJSON stdio protocol" requirement currently fixes token events as id-less and transport as private stdin/stdout; sharing adds token ids and the listen transport.

## Impact

- `Sources/` (connection handling, refcount, event emission), build/protocol tests, `README.md` (protocol contract), `openspec/specs/mlx-runner/spec.md`.
- Cross-repo: the TLC extension (`m4gshm/type-like-chuck-vs-code`, change `shared-stdio-runner`) connects instead of spawning and checks PID liveness; the openvino backend carries the same contract. Protocol stays backward compatible: stdin/stdout mode and id-less tokens keep working with old clients.
