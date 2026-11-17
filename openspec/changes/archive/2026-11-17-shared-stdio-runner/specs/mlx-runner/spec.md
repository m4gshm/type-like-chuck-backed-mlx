# Spec Delta

## MODIFIED Requirements

### Requirement: NDJSON stdio protocol
The runner SHALL read one request per stdin line as a JSON object (`generate` with `id`/`prompt`/`maxTokens`/`stop`, an optional `cache` flag defaulting to false, `cancel`, `shutdown`) and SHALL write newline-delimited JSON events to stdout: a `version` event for `--version`, one `token` event (with the request `id`) per generated text piece, one `done` event at the end of a generation carrying `promptTokens` (the tokenized prompt length) and `cachedTokens` (prompt tokens served from the retained KV cache; 0 without reuse), and `error` events on failures. Human-readable logs SHALL go to stderr only. With missing or invalid arguments the runner SHALL emit one `error` event and exit with code 2. When started with `--listen <endpoint>` the same protocol SHALL be served over a unix domain socket per the `runner-sharing` capability, with no TCP port and no API key.

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
- **THEN** it is queued and served after the active generation, in arrival order, and `token` events carry the `id` of the request being generated, so a multi-client (listen-mode) session never routes tokens to the wrong client

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
