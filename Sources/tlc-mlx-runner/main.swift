import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import Tokenizers
import os

struct InMessage: Decodable {
    let type: String
    let id: Int?
    let prompt: String?
    let maxTokens: Int?
    let stop: [String]?
    let cache: Bool?
    let token: String?
}

// Static methods on a Sendable enum are themselves Sendable, so these can be
// captured from @Sendable generation closures.
enum Wire {
    static func emit(_ payload: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: payload),
           let line = String(data: data, encoding: .utf8) {
            print(line)
            fflush(stdout)
        }
    }

    static func emit(to sink: Sink, _ payload: [String: Any]) {
        if !sink.writeLine(payload) {
            noteBrokenSink(sink)
        }
    }

    static func emitError(to sink: Sink, id: Int?, _ message: String) {
        var payload: [String: Any] = ["type": "error", "message": message]
        if let id {
            payload["id"] = id
        } else {
            payload["id"] = NSNull()
        }
        emit(to: sink, payload)
    }

    static func failNoModel(_ reason: String) -> Never {
        emitError(to: Sink(fd: 1), id: nil, reason)
        exit(2)
    }

    /// Emits only text that cannot be the beginning of a stop string; returns
    /// whether a full stop string was found (the caller then drops the tail).
    static func splitSafe(_ text: String, stops: [String]) -> (String, Bool) {
        var cut = Int.max
        var hit = false
        for stop in stops where !stop.isEmpty {
            if let range = text.range(of: stop) {
                let offset = text.distance(from: text.startIndex, to: range.lowerBound)
                if offset < cut { cut = offset }
                hit = true
            }
        }
        if hit { return (String(text.prefix(cut)), true) }
        let hold = (stops.map(\.count).max() ?? 1) - 1
        if hold > 0, text.count > hold {
            return (String(text.dropLast(hold)), false)
        }
        return (text, false)
    }
}

var arguments = Array(CommandLine.arguments.dropFirst())
var modelDir = ""
var ctx = 0
var printVersion = false
var listenPath = ""
var listenTcpPort = -1
var authToken = ""
var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--version":
        printVersion = true
    case "--model":
        index += 1
        guard index < arguments.count else { Wire.failNoModel("--model requires a value") }
        modelDir = arguments[index]
    case "--ctx":
        index += 1
        guard index < arguments.count, let value = Int(arguments[index]) else { Wire.failNoModel("--ctx requires a number") }
        ctx = value
    case "--listen":
        index += 1
        guard index < arguments.count else { Wire.failNoModel("--listen requires a value") }
        listenPath = arguments[index]
    case "--listen-tcp":
        index += 1
        guard index < arguments.count, let value = Int(arguments[index]), value >= 0, value <= 65535 else {
            Wire.failNoModel("--listen-tcp requires a port number (0 = random free port)")
        }
        listenTcpPort = value
    case "--token":
        index += 1
        guard index < arguments.count else { Wire.failNoModel("--token requires a value") }
        authToken = arguments[index]
    default:
        Wire.failNoModel("unknown argument \(arguments[index])")
    }
    index += 1
}
if printVersion {
    Wire.emit(["type": "version", "app": appVersion])
    exit(0)
}
guard !modelDir.isEmpty else { Wire.failNoModel("missing required --model") }

struct Request: Sendable {
    let id: Int
    let prompt: String
    let maxTokens: Int
    let stops: [String]
    let cache: Bool
    let sink: Sink
}

/// KV-cache reuse across generations: keeps the last prompt's cache and token
/// ids; a request whose prompt shares a long token prefix (every keystroke
/// extends the FIM prefix) continues from the cached rows instead of
/// re-prefilling them — the same trim-to-common-prefix move ChatSession makes.
/// Accessed only inside container.perform closures and by the generation task
/// strictly after its stream drained and before the next request starts; the
/// serial request queue makes that race-free.
final class PromptLedger: @unchecked Sendable {
    static let minTokens = 32

    private var cache: [any KVCache]?
    private var ids: [Int] = []
    private var usable = false

    /// Decides reuse for `tokens` and trims the cache to the shared prefix.
    /// Returns how many prompt tokens the kept cache already represents.
    func prepare(_ tokens: [Int], model: any LanguageModel, enabled: Bool) throws -> Int {
        var reuse = 0
        if enabled, usable, let current = cache, !current.isEmpty,
            current.allSatisfy({ $0.isTrimmable }) {
            reuse = min(Self.commonPrefix(ids, tokens), tokens.count - 1)
        }
        if reuse < Self.minTokens {
            reuse = 0
            cache = try model.newCache(parameters: nil)
        } else {
            for entry in cache! {
                let before = entry.offset
                if entry.trim(before - reuse) != before - reuse {
                    reuse = 0
                    cache = try model.newCache(parameters: nil)
                    break
                }
            }
        }
        return reuse
    }

    /// Adopts the cache a finished generation leaves behind. A generation that
    /// did not drain its stream (stop string, cancel, error) may still be
    /// appending rows in the background — `invalidate` keeps the ledger from
    /// trimming such a cache; the next request then rebuilds from scratch.
    func commit(ids: [Int]) {
        self.ids = ids
        usable = cache != nil
    }

    func invalidate() {
        usable = false
    }

    func cacheForRequest() -> [any KVCache]? {
        cache
    }

    private static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var i = 0
        while i < a.count && i < b.count && a[i] == b[i] { i += 1 }
        return i
    }
}

// Serial request queue served one generation at a time in arrival order —
// mirrors the ovgenai runner: `generate` while busy is queued, `cancel` of a
// queued request answers it immediately with done/`cancel`, and `shutdown`
// drains the queue the same way before cancelling the active generation.
// State is guarded by an async-safe unfair lock; the worker thread (a plain
// Thread, which may block freely) coordinates with the detached generation
// task through semaphores.
final class Worker: @unchecked Sendable {
    private struct State {
        var queue: [Request] = []
        var generation: Task<Void, Never>?
        var activeId: Int?
        var cancelledId: Int?
        var shuttingDown = false
    }

    private let state = OSAllocatedUnfairLock<State>(initialState: State())
    private let ledger = PromptLedger()
    /// Signals that a queued request is available.
    private let pending = DispatchSemaphore(value: 0)
    /// Signals that the worker loop has exited.
    private let loopDone = DispatchSemaphore(value: 0)

    /// Starts the worker thread serving queued requests in arrival order.
    func run(container: ModelContainer) {
        let thread = Thread { [weak self] in self?.loop(container: container) }
        thread.name = "runner-worker"
        thread.start()
    }

    func enqueue(_ request: Request) {
        state.withLock { $0.queue.append(request) }
        pending.signal()
    }

    func cancel(_ id: Int?, sink: Sink) {
        let removedId: Int? = state.withLock { st in
            if let id {
                if id == st.activeId {
                    st.cancelledId = id
                    st.generation?.cancel()
                } else if let index = st.queue.firstIndex(where: { $0.id == id }) {
                    // A cancelled queued request is answered immediately.
                    st.queue.remove(at: index)
                    return id
                }
            } else if st.activeId != nil {
                // cancel without id targets the active generation.
                st.cancelledId = st.activeId
                st.generation?.cancel()
            }
            return nil
        }
        if let removedId {
            Wire.emit(to: sink, ["type": "done", "id": removedId, "finishReason": "cancel"])
        }
    }

    /// Drains the queue (every pending request answered with done/`cancel`),
    /// cancels the active generation and waits for the worker to finish.
    func shutdown(sink: Sink) {
        let dropped: [Int] = state.withLock { st in
            st.shuttingDown = true
            let dropped = st.queue.map(\.id)
            st.queue.removeAll()
            if let active = st.activeId {
                st.cancelledId = active
                st.generation?.cancel()
            }
            return dropped
        }
        pending.signal()
        loopDone.wait()
        for id in dropped {
            Wire.emit(to: sink, ["type": "done", "id": id, "finishReason": "cancel"])
        }
    }

    private func loop(container: ModelContainer) {
        while true {
            pending.wait()
            // nil = exit (shut down and fully drained); a request with a nil
            // task marker means the item was drained elsewhere — re-check.
            let next: Request?? = state.withLock { st -> Request?? in
                if st.shuttingDown && st.queue.isEmpty && st.generation == nil { return nil }
                return st.queue.isEmpty ? .some(nil) : .some(st.queue.removeFirst())
            }
            guard let outer = next else { break }
            guard let request = outer else { continue }

            state.withLock { st in
                st.activeId = request.id
                st.cancelledId = nil
            }
            // Detached: never inherit the main actor. The loop thread blocks
            // on `finished` until the generation completes — requests are
            // served strictly one at a time.
            let finished = DispatchSemaphore(value: 0)
            let generation = Task.detached { [weak self] in
                await self?.generate(request, container: container)
                self?.state.withLock { st in
                    st.generation = nil
                    st.activeId = nil
                    st.cancelledId = nil
                }
                finished.signal()
            }
            state.withLock { $0.generation = generation }
            finished.wait()
        }
        // Every request awaited its producer task in generate() — including
        // the producer's final GPU synchronize — so no MLX work outlives the
        // loop and process teardown cannot race it.
        loopDone.signal()
    }

    private func generate(_ request: Request, container: ModelContainer) async {
        var finishReason = "stop"
        // cancel/shutdown may flag the request in the window before its
        // generation task is created — nothing was there to cancel(), so
        // resolve it here instead of generating.
        if state.withLock({ $0.cancelledId == request.id }) {
            Wire.emit(to: request.sink, ["type": "done", "id": request.id, "finishReason": "cancel"])
            return
        }
        var promptTokens = 0
        var cachedTokens = 0
        do {
            // Prepare inside the container's isolation, hand the sendable
            // stream and producer task out and consume the stream here — the
            // loop drives itself in its own task.
            let (stream, producer, tokens, reused, tokenizer) = try await container.perform { context in
                // Raw tokenization, no chat template: the plugin sends fully
                // rendered FIM prompts; wrapping them in a chat breaks base
                // models and deflects instruct ones.
                let tokens = context.tokenizer.encode(text: request.prompt, addSpecialTokens: false)
                let reused = try ledger.prepare(tokens, model: context.model, enabled: request.cache)
                let input = reused > 0
                    ? LMInput(text: .init(tokens: MLXArray(Array(tokens[reused...]))))
                    : LMInput(text: .init(tokens: MLXArray(tokens)))
                // Text LLMs here (Qwen2.5) carry no per-call positional state,
                // so a trimmed cache needs no carried state to reanchor rows.
                let (stream, producer) = try MLXLMCommon.generateTokensTask(
                    input: input,
                    cache: ledger.cacheForRequest(),
                    parameters: GenerateParameters(maxTokens: request.maxTokens),
                    context: context)
                return (stream, producer, tokens, reused, context.tokenizer)
            }
            promptTokens = tokens.count
            cachedTokens = reused
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            var buffer = ""
            var stopHit = false
            events: for await event in stream {
                switch event {
                case .token(let token):
                    detokenizer.append(token: token)
                    if let piece = detokenizer.next() {
                        buffer += piece
                        let (safe, hit) = Wire.splitSafe(buffer, stops: request.stops)
                        if !safe.isEmpty {
                            Wire.emit(to: request.sink, ["type": "token", "id": request.id, "text": safe])
                            buffer = String(buffer.dropFirst(safe.count))
                        }
                        if hit {
                            stopHit = true
                            finishReason = "stop"
                            break events
                        }
                    }
                case .info(let info):
                    switch info.stopReason {
                    case .length where !stopHit: finishReason = "length"
                    case .cancelled: finishReason = "cancel"
                    default: break
                    }
                }
            }
            // Awaiting the producer settles its final GPU synchronize: after
            // this nothing outlives the request and exit cannot race Metal
            // teardown.
            await producer.value
            let (safe, _) = Wire.splitSafe(buffer, stops: request.stops)
            if !safe.isEmpty { Wire.emit(to: request.sink, ["type": "token", "id": request.id, "text": safe]) }
            // A cancelled task ends the stream without an .info event.
            if Task.isCancelled && !stopHit { finishReason = "cancel" }
            if stopHit || finishReason == "cancel" {
                ledger.invalidate()
            } else {
                ledger.commit(ids: tokens)
            }
        } catch is CancellationError {
            ledger.invalidate()
            finishReason = "cancel"
        } catch {
            ledger.invalidate()
            let wasCancelled = state.withLock { $0.cancelledId == request.id }
            if wasCancelled {
                finishReason = "cancel"
            } else {
                // The ovgenai contract reports a failed generation with a
                // single error event — no done event follows.
                Wire.emitError(to: request.sink, id: request.id, "generate failed: \(error)")
                return
            }
        }
        Wire.emit(to: request.sink, ["type": "done", "id": request.id, "finishReason": finishReason,
                   "promptTokens": promptTokens, "cachedTokens": cachedTokens])
    }
}

let worker = Worker()

let container: ModelContainer
do {
    container = try await LLMModelFactory.shared.loadContainer(
        from: URL(filePath: modelDir), using: #huggingFaceTokenizerLoader())
} catch {
    Wire.emitError(to: Sink(fd: 1), id: nil, "model load failed: \(error)")
    exit(1)
}

signal(SIGPIPE, SIG_IGN)

var listenFd: Int32?
var listenIsTcp = false
var listenPort: UInt16?
if !listenPath.isEmpty {
    do {
        listenFd = try listenUnix(path: listenPath)
    } catch ListenFailure.busy {
        stderrLog("listen endpoint already in use: \(listenPath)")
        Wire.emitError(to: Sink(fd: 1), id: nil, "listen endpoint already in use: \(listenPath)")
        exit(1)
    } catch {
        stderrLog("listen setup failed: \(error)")
        Wire.emitError(to: Sink(fd: 1), id: nil, "listen setup failed: \(listenPath)")
        exit(2)
    }
} else if listenTcpPort >= 0 {
    do {
        let (fd, port) = try listenTcp(port: UInt16(listenTcpPort))
        listenFd = fd
        listenPort = port
        listenIsTcp = true
    } catch ListenFailure.busy {
        stderrLog("listen port already in use: \(listenTcpPort)")
        Wire.emitError(to: Sink(fd: 1), id: nil, "listen port already in use: \(listenTcpPort)")
        exit(1)
    } catch {
        stderrLog("listen setup failed: \(error)")
        Wire.emitError(to: Sink(fd: 1), id: nil, "listen setup failed on port \(listenTcpPort)")
        exit(2)
    }
}

var readyLine: [String: Any] = [
    "type": "ready",
    "model": modelDir,
    "device": "gpu",
    "ctx": ctx,
    "pid": ProcessInfo.processInfo.processIdentifier,
]
if let listenPort { readyLine["port"] = Int(listenPort) }

func broadcastReady(to sink: Sink) {
    Wire.emit(to: sink, readyLine)
}

let stdinClient = Client(fd: 0, needsAuth: false, label: "stdio client")
clients.add(stdinClient)
registerSinkOwner(stdinClient.sink, client: stdinClient)
Wire.emit(to: stdinClient.sink, readyLine)
worker.run(container: container)

clients.onLastClientLeft = { reason in
    stderrLog("self-exit: \(reason)")
    Thread(block: {
        worker.shutdown(sink: discardSink)
        exit(0)
    }).start()
}

func handleLine(_ line: String, client: Client) {
    guard let data = line.data(using: .utf8),
          let message = try? JSONDecoder().decode(InMessage.self, from: data) else {
        Wire.emitError(to: client.sink, id: nil, "unparseable request")
        return
    }
    // The auth gate applies to TCP only: a localhost port is reachable by
    // any local user, a unix socket and the spawner's stdin are not.
    if client.needsAuth && !client.isAuthenticated() {
        if message.type == "auth", let token = message.token, client.tryAuth(token, against: authToken) {
            return
        }
        Wire.emitError(to: client.sink, id: nil, "authentication required: send {\"type\":\"auth\",\"token\":...} first")
        dropClient(client, reason: "authentication failed")
        return
    }
    switch message.type {
    case "auth":
        break
    case "generate":
        guard let id = message.id, let prompt = message.prompt else {
            Wire.emitError(to: client.sink, id: message.id, "generate requires id and prompt")
            return
        }
        worker.enqueue(Request(
            id: id,
            prompt: prompt,
            maxTokens: message.maxTokens ?? 128,
            stops: message.stop ?? [],
            cache: message.cache ?? false,
            sink: client.sink))
    case "cancel":
        worker.cancel(message.id, sink: client.sink)
    case "shutdown":
        worker.shutdown(sink: client.sink)
        exit(0)
    default:
        Wire.emitError(to: client.sink, id: message.id, "unknown request type \(message.type)")
    }
}

if let listenFd {
    serveListener(fd: listenFd, isTcp: listenIsTcp, handleLine: handleLine)
}

// Blocking line reads on a dedicated thread: request handling is synchronous
// (generation runs in its own detached task), so a read blocked on stdin
// never delays an in-flight generation — mirroring the ovgenai runner. The
// spawner's stdio counts as a client even in listen mode: a stdin nobody
// read silently froze the spawning window's completions (openvino pitfall).
spawnThread {
    let reader = LineReader(fd: 0)
    while let line = reader.nextLine() {
        handleLine(line, stdinClient)
    }
    dropClient(stdinClient, reason: "stdin closed")
}
dispatchMain()
