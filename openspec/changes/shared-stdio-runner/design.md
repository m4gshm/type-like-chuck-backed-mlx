# Design

## Context

The runner serves NDJSON over stdin/stdout (Swift/Sources), requests queue in arrival order, tokens are emitted id-less, and the extension owns the process lifetime. The OpenVINO runner repo carries the identical protocol and gets the same contract (its own `shared-stdio-runner` change); the extension repo has the client-side change.

## Goals / Non-Goals

**Goals:** one runner per model across windows on one Mac; runner lifetime bound to connected clients; backward-compatible protocol.

**Non-Goals:** TCP or remote endpoints; concurrent generation; authentication (unix socket permissions are user-scoped by default).

## Decisions

### Transport: unix domain socket by default, TCP for IPC boundaries
`--listen` uses a socket path; `--listen-tcp <port|0>` adds a `127.0.0.1` TCP transport for setups where local IPC does not cross the plugin/backend boundary (VS Code remote). Port `0` binds a random free port and reports it on the handshake, so the spawning extension can publish it in its discovery file. The endpoint/port value is computed by the extension, the runner treats it as opaque. Same NDJSON framing on accepted connections; the event loop is parameterized over the line source (stdin or connections). Ownership of a socket path is checked bind-then-probe (an unconditional unlink would hijack a live runner's endpoint).

Alternative considered: localhost TCP — rejected: violates the "no network port" invariant and needs auth.

### Refcount lives inside the runner
Count open connections; on 1→0 transition (after having been ≥1) exit cleanly with the active generation cancelled. A failed client write counts as disconnect. No lock files, no owner election.

### Token ids
Add `id` to every `token` event. Old clients ignore the extra field; the new extension falls back to oldest-pending routing when talking to an old runner. Queue semantics unchanged — one generation at a time.

### PID in the handshake
The version/ready handshake event gains `pid`, used by the extension for liveness pre-checks and status display.

## Risks / Trade-offs

- [Backend and extension land out of order] → both directions are compatible: old runner + new extension = private spawn fallback; new runner + old extension = unchanged behavior.
- [TCP on localhost without auth] → opt-in only, for IPC-boundary setups; a foreign local process could generate completions but cannot read anything private. Risk accepted; noted for review.
- [Instance discovery lives outside the runner] → per-runner discovery files and option matching (model only for mlx: MLX compiles once, dynamic KV cache — windows with different context lengths share one runner) are the extension's responsibility; the runner only reports pid/port on the handshake.
- [Second runner on the same socket path] → bind fails; runner exits with an `error` event; the extension's connect-or-spawn falls back to its own private runner.
- [Prompt KV cache with multiple clients] → cache reuse stays tied to the previous cleanly finished generation regardless of which client it came from; acceptable for completion workloads, noted in the spec delta (cache scenarios unchanged).

## Migration Plan

1. Land listen mode + refcount + token ids here; publish a new `v*` release.
2. The extension imports it via `RUNNER_VERSION` and enables connect-or-spawn.
3. Rollback: the extension setting `tlc.local.sharedRunner: false` restores private-pipe spawns without any runner change.
