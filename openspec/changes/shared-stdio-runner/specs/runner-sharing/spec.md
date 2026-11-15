# Spec Delta

## ADDED Requirements

### Requirement: Listen mode
The runner SHALL accept `--listen <endpoint>` to serve its NDJSON protocol over a unix domain socket, and `--listen-tcp <port>` (port `0` = a random free localhost port, the chosen port reported on the handshake) to serve it over `127.0.0.1:<port>` — without requiring an API key. Requests, events, ordering, cancellation, cache reuse and shutdown semantics SHALL be identical on all transports. The listen endpoint MAY combine with `--model` and the usual start options; readiness is reported on the connection as today. The runner SHALL NOT read or write any instance-discovery files; that is the client's responsibility.

#### Scenario: Client connects and generates
- **WHEN** the runner starts with `--listen <socket>` and a client connects and sends a `generate` request
- **THEN** the client receives the same `token`/`done` event stream as a stdin/stdout client would

#### Scenario: TCP with a random port
- **WHEN** the runner starts with `--listen-tcp 0` and a client connects to the reported port on `127.0.0.1`
- **THEN** the client receives the same `token`/`done` event stream, and the handshake carries the chosen port

### Requirement: Client refcount and self-exit
In listen mode the runner SHALL track its connected clients and exit by itself when the connection count returns to zero after having been at least one. A client write error or protocol timeout SHALL count as a disconnect. The runner SHALL NOT exit before the first client connects.

#### Scenario: Last client disconnects
- **WHEN** the only connected client closes its connection
- **THEN** the runner exits by itself without waiting for a `shutdown` request

#### Scenario: Runner outlives its spawner
- **WHEN** the client that started the runner disconnects while another client is still connected
- **THEN** the runner keeps running and serving the remaining client

#### Scenario: Hung client treated as disconnected
- **WHEN** a connected client stops reading and the runner's writes to it fail
- **THEN** the runner drops that client and updates the connection count as if it had disconnected

### Requirement: Per-request token routing
Every `token` event SHALL carry the `id` of the request it belongs to, so that multiple connected clients each receive only the tokens of their own requests. Generation is still served one request at a time in arrival order; the id does not enable concurrent generation.

#### Scenario: Tokens carry the request id
- **WHEN** a client sends a `generate` request with `id` 7 and the runner streams the completion
- **THEN** every `token` event of that generation carries `id` 7, followed by the `done` event with `id` 7

#### Scenario: Two clients do not cross streams
- **WHEN** two clients send generate requests and the runner serves them in arrival order
- **THEN** each client's stream contains only the token events of its own request

#### Scenario: Stdin/stdout compatibility
- **WHEN** a legacy client sends a `generate` request without listening to ids
- **THEN** the token stream is unchanged apart from the added `id` field, which legacy clients ignore

### Requirement: PID and port exposure on ready
The runner SHALL include its process id (`pid`) — and, when started with `--listen-tcp 0`, the chosen localhost port (`port`) — in its startup handshake event (the `version`/ready event), so clients can validate backend liveness before connecting, record how to connect, and identify the process in status output.

#### Scenario: Handshake carries the pid
- **WHEN** the runner reports readiness to a connected or piped client
- **THEN** the handshake event includes the runner's own OS process id

#### Scenario: Handshake carries the chosen TCP port
- **WHEN** the runner starts with `--listen-tcp 0` and reports readiness
- **THEN** the handshake event includes the actual bound port
