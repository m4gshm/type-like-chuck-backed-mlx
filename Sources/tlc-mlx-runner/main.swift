import Foundation
import MLXLMCommon
import MLXLLM
struct InMessage: Decodable {
    let type: String
    let id: Int?
    let prompt: String?
    let maxTokens: Int?
    let stop: [String]?
}

func emit(_ payload: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: payload),
       let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }
}

func emitError(id: Int?, _ message: String) {
    emit(["type": "error", "id": id as Any?, "message": message])
}

func failNoModel(_ reason: String) -> Never {
    emitError(id: nil, reason)
    exit(2)
}

var arguments = Array(CommandLine.arguments.dropFirst())
var modelDir = ""
var ctx = 0
var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--model":
        index += 1
        guard index < arguments.count else { failNoModel("--model requires a value") }
        modelDir = arguments[index]
    case "--ctx":
        index += 1
        guard index < arguments.count, let value = Int(arguments[index]) else { failNoModel("--ctx requires a number") }
        ctx = value
    default:
        failNoModel("unknown argument \(arguments[index])")
    }
    index += 1
}
guard !modelDir.isEmpty else { failNoModel("missing required --model") }

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

final class Worker: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<Void, Never>?
    private var cancelledId: Int?

    func busy() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return task != nil
    }

    func start(id: Int, prompt: String, maxTokens: Int, stops: [String], container: ModelContainer) {
        let generation = Task {
            var finishReason = "stop"
            do {
                try await container.perform { context in
                    let input = try context.processor.prepare(input: UserInput(prompt: prompt))
                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
                    var buffer = ""
                    var tokenCount = 0
                    _ = try MLXLMCommon.generate(input: input, parameters: GenerateParameters(maxTokens: maxTokens), context: context) { tokens in
                        tokenCount = tokens.count
                        if let last = tokens.last {
                            detokenizer.append(token: last)
                        }
                        while let piece = detokenizer.next() {
                            buffer += piece
                            let (safe, hit) = splitSafe(buffer, stops: stops)
                            if !safe.isEmpty {
                                emit(["type": "token", "id": id, "text": safe])
                                buffer = String(buffer.dropFirst(safe.count))
                            }
                            if hit {
                                buffer = ""
                                return .stop
                            }
                        }
                        return tokenCount >= maxTokens ? .stop : .more
                    }
                    let (safe, hit) = splitSafe(buffer, stops: stops)
                    if !safe.isEmpty { emit(["type": "token", "id": id, "text": safe]) }
                    if tokenCount >= maxTokens && !hit { finishReason = "length" }
                }
            } catch is CancellationError {
                finishReason = "cancel"
            } catch {
                lock.lock()
                let wasCancelled = cancelledId == id
                lock.unlock()
                if wasCancelled {
                    finishReason = "cancel"
                } else {
                    emitError(id: id, "\(error)")
                    emit(["type": "done", "id": id, "finishReason": "error"])
                    self.clearTask()
                    return
                }
            }
            emit(["type": "done", "id": id, "finishReason": finishReason])
            self.clearTask()
        }
        lock.lock()
        task = generation
        lock.unlock()
    }

    func cancel(id: Int?) {
        lock.lock()
        cancelledId = id
        task?.cancel()
        lock.unlock()
    }

    private func clearTask() {
        lock.lock()
        task = nil
        cancelledId = nil
        lock.unlock()
    }
}

let worker = Worker()
let modelName = URL(filePath: modelDir).lastPathComponent

let container: ModelContainer
do {
    let configuration = ModelConfiguration(directory: URL(filePath: modelDir))
    container = try await LLMModelFactory.shared.loadContainer(configuration: configuration)
} catch {
    emitError(id: nil, "model load failed: \(error)")
    exit(1)
}
emit(["type": "ready", "model": modelName, "device": "gpu", "ctx": ctx])

for try await line in FileHandle.standardInput.bytes.lines {
    guard let data = line.data(using: .utf8),
          let message = try? JSONDecoder().decode(InMessage.self, from: data) else {
        emitError(id: nil, "unparseable request")
        continue
    }
    switch message.type {
    case "generate":
        guard let id = message.id, let prompt = message.prompt else {
            emitError(id: message.id, "generate requires id and prompt")
            continue
        }
        if worker.busy() {
            emitError(id: id, "busy")
            continue
        }
        worker.start(
            id: id,
            prompt: prompt,
            maxTokens: message.maxTokens ?? 128,
            stops: message.stop ?? [],
            container: container
        )
    case "cancel":
        worker.cancel(id: message.id)
    case "shutdown":
        exit(0)
    default:
        emitError(id: message.id, "unknown request type \(message.type)")
    }
}
exit(0)
