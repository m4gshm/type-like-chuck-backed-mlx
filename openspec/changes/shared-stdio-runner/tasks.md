# Tasks

## 1. Listen mode

- [ ] 1.1 Parse `--listen <endpoint>` and `--listen-tcp <port|0>` (arg tests: missing value, invalid path, stale socket file handling) and verify the test suite passes
- [ ] 1.2 Implement the unix domain socket server (bind-then-probe ownership check before any unlink) and the localhost TCP server (port 0 = random free port), parameterizing the NDJSON loop over the line source (stdin or connection); verify with a manual two-connection smoke script
- [ ] 1.3 Report failure on a taken endpoint (`error` event, non-zero exit) and verify with a unit or manual test

## 2. Client refcount and self-exit

- [ ] 2.1 Track open connections; exit cleanly (cancel active generation with `finishReason:"cancel"`) when the count returns to zero after having been ≥1; verify by connecting and dropping clients in a scripted smoke test
- [ ] 2.2 Treat a failed/blocked client write as a disconnect; verify with a client that stops reading

## 3. Token ids and PID

- [ ] 3.1 Add the request `id` to every `token` event; keep stdin/stdout behavior otherwise identical; verify the protocol tests
- [ ] 3.2 Add `pid` (and the chosen `port` with `--listen-tcp 0`) to the startup handshake event and verify it in the protocol e2e

## 4. Contract and release

- [ ] 4.1 Update `README.md` protocol section (token `id`, handshake `pid`, `--listen`, refcount self-exit) and the spec via this change's delta on archive
- [ ] 4.2 Keep the contract in sync with the openvino backend repo (`type-like-chuck-backed-openvino`, same change name); verify by diffing the protocol sections of both READMEs
- [ ] 4.3 Run `task build` and the protocol e2e; publish a `v*` release so the extension can pin it via `RUNNER_VERSION`
