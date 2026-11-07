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

    static func emitError(id: Int?, _ message: String) {
        var payload: [String: Any] = ["type": "error", "message": message]
        if let id {
            payload["id"] = id
        } else {
            payload["id"] = NSNull()
        }
        emit(payload)
    }

    static func failNoModel(_ reason: String) -> Never {
        emitError(id: nil, reason)
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

// Emits only text that cannot be the beginning of a stop string; returns
// whether a full stop string was found (the caller then drops the tail).
func splitSafe(_ text: String, stops: [String]) -> (String, Bool) {
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

struct Request: Sendable {
    let id: Int
    let prompt: String
    let maxTokens: Int
    let stops: [String]
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

    func cancel(_ id: Int?) {
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
            Wire.emit(["type": "done", "id": removedId, "finishReason": "cancel"])
        }
    }

    /// Drains the queue (every pending request answered with done/`cancel`),
    /// cancels the active generation and waits for the worker to finish.
    func shutdown() {
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
            Wire.emit(["type": "done", "id": id, "finishReason": "cancel"])
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
        // The last cancelled MLX generation unwinds on its own threads (Metal
        // command buffers, the generate loop's executor); exiting while that
        // teardown is still running segfaults. Wait it out here — this is a
        // plain Thread, so blocking sleep is fine.
        Thread.sleep(forTimeInterval: 0.5)
        loopDone.signal()
    }

    private func generate(_ request: Request, container: ModelContainer) async {
        var finishReason = "stop"
        // cancel/shutdown may flag the request in the window before its
        // generation task is created — nothing was there to cancel(), so
        // resolve it here instead of generating.
        if state.withLock({ $0.cancelledId == request.id }) {
            Wire.emit(["type": "done", "id": request.id, "finishReason": "cancel"])
            return
        }
        do {
            // Prepare inside the container's isolation, hand the sendable
            // stream out and consume it here — the loop drives itself in
            // its own task.
            let stream: AsyncStream<Generation> = try await container.perform { context in
                // Raw tokenization, no chat template: the plugin sends fully
                // rendered FIM prompts; wrapping them in a chat breaks base
                // models and deflects instruct ones.
                let tokens = context.tokenizer.encode(text: request.prompt, addSpecialTokens: false)
                let input = LMInput(text: .init(tokens: MLXArray(tokens)))
                return try MLXLMCommon.generate(
                    input: input,
                    parameters: GenerateParameters(maxTokens: request.maxTokens),
                    context: context)
            }
            var buffer = ""
            var stopHit = false
            events: for await event in stream {
                switch event {
                case .chunk(let piece):
                    buffer += piece
                    let (safe, hit) = Wire.splitSafe(buffer, stops: request.stops)
                    if !safe.isEmpty {
                        Wire.emit(["type": "token", "text": safe])
                        buffer = String(buffer.dropFirst(safe.count))
                    }
                    if hit {
                        stopHit = true
                        finishReason = "stop"
                        break events
                    }
                case .info(let info):
                    switch info.stopReason {
                    case .length where !stopHit: finishReason = "length"
                    case .cancelled: finishReason = "cancel"
                    default: break
                    }
                case .toolCall, .rejectedToolCall:
                    break
                }
            }
            let (safe, _) = Wire.splitSafe(buffer, stops: request.stops)
            if !safe.isEmpty { Wire.emit(["type": "token", "text": safe]) }
            // A cancelled task ends the stream without an .info event.
            if Task.isCancelled && !stopHit { finishReason = "cancel" }
        } catch is CancellationError {
            finishReason = "cancel"
        } catch {
            let wasCancelled = state.withLock { $0.cancelledId == request.id }
            if wasCancelled {
                finishReason = "cancel"
            } else {
                // The ovgenai contract reports a failed generation with a
                // single error event — no done event follows.
                Wire.emitError(id: request.id, "generate failed: \(error)")
                return
            }
        }
        Wire.emit(["type": "done", "id": request.id, "finishReason": finishReason])
    }
}

let worker = Worker()

let container: ModelContainer
do {
    container = try await LLMModelFactory.shared.loadContainer(
        from: URL(filePath: modelDir), using: #huggingFaceTokenizerLoader())
} catch {
    Wire.emitError(id: nil, "model load failed: \(error)")
    exit(1)
}
Wire.emit(["type": "ready", "model": modelDir, "device": "gpu", "ctx": ctx])
worker.run(container: container)

// Blocking line reads on the main thread: request handling is synchronous
// (generation runs in its own detached task), so a read blocked on stdin
// never delays an in-flight generation — mirroring the ovgenai runner.
while true {
    guard let line = readLine() else { break }
    guard let data = line.data(using: .utf8),
          let message = try? JSONDecoder().decode(InMessage.self, from: data) else {
        Wire.emitError(id: nil, "unparseable request")
        continue
    }
    switch message.type {
    case "generate":
        guard let id = message.id, let prompt = message.prompt else {
            Wire.emitError(id: message.id, "generate requires id and prompt")
            continue
        }
        worker.enqueue(Request(
            id: id,
            prompt: prompt,
            maxTokens: message.maxTokens ?? 128,
            stops: message.stop ?? []))
    case "cancel":
        worker.cancel(message.id)
    case "shutdown":
        worker.shutdown()
        exit(0)
    default:
        Wire.emitError(id: message.id, "unknown request type \(message.type)")
    }
}
worker.shutdown()
exit(0)
