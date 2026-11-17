import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Line-oriented output to a client: fd 1 for the spawner's stdio, an
/// accepted socket fd otherwise. A failed write marks the sink broken — the
/// owner drops the client, mirroring the ovgenai runner's sinkBroken path.
/// fd -1 discards output (used where events have nobody left to reach).
final class Sink: @unchecked Sendable {
    let fd: Int32
    private let lock = NSLock()

    init(fd: Int32) {
        self.fd = fd
    }

    func writeLine(_ payload: [String: Any]) -> Bool {
        if fd < 0 { return false }
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return false }
        var bytes = [UInt8](data)
        bytes.append(0x0A)
        lock.lock()
        defer { lock.unlock() }
        var offset = 0
        while offset < bytes.count {
            let n = bytes.withUnsafeBytes { buffer -> Int in
                write(fd, buffer.baseAddress!.advanced(by: offset), bytes.count - offset)
            }
            if n <= 0 { return false }
            offset += n
        }
        return true
    }
}

final class Client: @unchecked Sendable {
    let sink: Sink
    /// Only TCP clients authenticate: a unix socket lives in the user's
    /// namespace and the spawner's stdin is inherited by trust.
    let needsAuth: Bool
    let label: String
    private let lock = NSLock()
    private var authenticated = false

    init(fd: Int32, needsAuth: Bool, label: String) {
        self.sink = Sink(fd: fd)
        self.needsAuth = needsAuth
        self.label = label
    }

    func tryAuth(_ token: String, against expected: String) -> Bool {
        guard !expected.isEmpty, token == expected else { return false }
        lock.lock()
        authenticated = true
        lock.unlock()
        return true
    }

    func isAuthenticated() -> Bool {
        lock.withLock { authenticated }
    }
}

/// Refcount of open clients (stdin counts as one). When the count returns to
/// zero after having been ≥1 the runner exits by itself — closing a window
/// never kills a runner other windows still use.
final class Clients: @unchecked Sendable, CustomStringConvertible {
    private let lock = NSLock()
    private var set: Set<ObjectIdentifier> = []
    private var everHadClient = false
    private var exiting = false
    var onLastClientLeft: ((String) -> Void)?

    var description: String {
        "total \(lock.withLock { set.count })"
    }

    func add(_ client: Client) {
        lock.lock()
        everHadClient = true
        set.insert(ObjectIdentifier(client))
        let total = set.count
        lock.unlock()
        stderrLog("\(client.label) connected (total \(total))")
    }

    func drop(_ client: Client, reason: String) {
        lock.lock()
        guard !exiting, set.contains(ObjectIdentifier(client)) else {
            lock.unlock()
            return
        }
        set.remove(ObjectIdentifier(client))
        let total = set.count
        let last = total == 0 && everHadClient
        if last { exiting = true }
        lock.unlock()
        close(client.sink.fd)
        stderrLog("\(client.label) disconnected: \(reason) (total \(total))")
        if last {
            FileHandle.standardError.write(Data("[runner] exit requested: last client left\n".utf8))
            onLastClientLeft?("last client left")
        }
    }
}

let clients = Clients()

/// Events with nobody left to reach (the self-exit drain) go here.
let discardSink = Sink(fd: -1)

final class SinkOwnerRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var owners: [ObjectIdentifier: Client] = [:]

    func register(_ sink: Sink, client: Client) {
        lock.withLock { owners[ObjectIdentifier(sink)] = client }
    }

    func owner(_ sink: Sink) -> Client? {
        lock.withLock { owners[ObjectIdentifier(sink)] }
    }
}

let sinkOwners = SinkOwnerRegistry()

/// A failed client write is a disconnect: drop via the registry so the
/// refcount drives the self-exit.
func noteBrokenSink(_ sink: Sink) {
    if let owner = sinkOwners.owner(sink) {
        dropClient(owner, reason: "broken sink")
    }
}

func dropClient(_ client: Client, reason: String) {
    clients.drop(client, reason: reason)
}

/// Buffered line reads off a blocking fd, one reader thread per client.
final class LineReader {
    private let fd: Int32
    private var buffer: [UInt8] = []

    init(fd: Int32) {
        self.fd = fd
    }

    func nextLine() -> String? {
        while true {
            if let nl = buffer.firstIndex(of: 0x0A) {
                let line = String(decoding: buffer[0..<nl], as: UTF8.self)
                buffer.removeSubrange(0...nl)
                return line
            }
            var chunk = [UInt8](repeating: 0, count: 4096)
            let n = chunk.withUnsafeMutableBytes { read(fd, $0.baseAddress, $0.count) }
            if n <= 0 { return nil }
            buffer.append(contentsOf: chunk[0..<n])
        }
    }
}

enum ListenFailure: Error {
    case busy(String)
    case failed(String)
}

/// Unix socket with bind-then-probe ownership: connect to an existing path
/// first — a live listener means the endpoint is taken (exit 1); a refused
/// connect means a stale file after a crash, safe to unlink and rebind.
func listenUnix(path: String) throws -> Int32 {
    if unixEndpointAlive(path) {
        throw ListenFailure.busy(path)
    }
    unlink(path)
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { throw ListenFailure.failed("socket: errno \(errno)") }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(path.utf8CString)
    guard bytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
        close(fd)
        throw ListenFailure.failed("listen path too long: \(path)")
    }
    withUnsafeMutableBytes(of: &addr.sun_path) { dst in
        bytes.withUnsafeBytes { src in
            memcpy(dst.baseAddress, src.baseAddress, src.count)
        }
    }
    let bindResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
            bind(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard bindResult == 0 else {
        close(fd)
        throw ListenFailure.busy(path)
    }
    guard listen(fd, 8) == 0 else {
        close(fd)
        throw ListenFailure.failed("listen: errno \(errno)")
    }
    return fd
}

private func unixEndpointAlive(_ path: String) -> Bool {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(path.utf8CString)
    guard bytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else { return false }
    withUnsafeMutableBytes(of: &addr.sun_path) { dst in
        bytes.withUnsafeBytes { src in
            memcpy(dst.baseAddress, src.baseAddress, src.count)
        }
    }
    return withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
            connect(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0
        }
    }
}

/// Localhost TCP listener; port 0 picks a random free port, reported on the
/// ready event so the spawner can publish it in the discovery file.
func listenTcp(port: UInt16) throws -> (fd: Int32, port: UInt16) {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw ListenFailure.failed("socket: errno \(errno)") }
    var yes: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr = in_addr(s_addr: INADDR_ANY)
    let bindResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
            bind(fd, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindResult == 0 else {
        close(fd)
        throw ListenFailure.busy("tcp port \(port)")
    }
    guard listen(fd, 8) == 0 else {
        close(fd)
        throw ListenFailure.failed("listen: errno \(errno)")
    }
    var bound = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    let nameResult = withUnsafeMutablePointer(to: &bound) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
            getsockname(fd, sa, &len)
        }
    }
    guard nameResult == 0 else {
        close(fd)
        throw ListenFailure.failed("getsockname: errno \(errno)")
    }
    return (fd, UInt16(bigEndian: bound.sin_port))
}

/// Accept loop + per-client reader thread; `handleLine` is the shared
/// dispatcher also used for stdin.
func serveListener(fd: Int32, isTcp: Bool, handleLine: @escaping @Sendable (String, Client) -> Void) {
    spawnThread {
        while true {
            let clientFd = accept(fd, nil, nil)
            guard clientFd >= 0 else { break }
            let client = Client(fd: clientFd, needsAuth: isTcp, label: "client \(clientFd)")
            clients.add(client)
            sinkOwners.register(client.sink, client: client)
            broadcastReady(to: client.sink)
            spawnThread {
                let reader = LineReader(fd: clientFd)
                while let line = reader.nextLine() {
                    handleLine(line, client)
                }
                dropClient(client, reason: "closed")
            }
        }
    }
}

func spawnThread(_ block: @escaping @Sendable () -> Void) {
    Thread(block: block).start()
}

func stderrLog(_ text: String) {
    FileHandle.standardError.write(Data(("[runner] " + text + "\n").utf8))
}
