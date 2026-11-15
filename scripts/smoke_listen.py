#!/usr/bin/env python3
# Two-client smoke for the mlx listen mode (run on macOS arm64 with a local model):
#   task build && scripts/smoke_listen.py <model-dir>
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

runner = "build/tlc-mlx-runner-macos-arm64"
model = sys.argv[1] if len(sys.argv) > 1 else None
if not model:
    sys.exit("usage: scripts/smoke_listen.py <model-dir>")
token = "s3cret"


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def read_event(f, expect_type=None):
    ev = json.loads(f.readline())
    if expect_type and ev["type"] != expect_type:
        fail(f"expected {expect_type}, got {ev}")
    return ev


def generate(f, req_id, prompt, max_tokens=16):
    f.write(json.dumps({"type": "generate", "id": req_id, "prompt": prompt,
                        "maxTokens": max_tokens, "stop": ["\n"]}) + "\n")
    f.flush()
    tokens = []
    while True:
        ev = read_event(f)
        if ev["type"] == "token":
            if ev.get("id") != req_id:
                fail(f"wrong token id: {ev}")
            tokens.append(ev["text"])
        elif ev["type"] == "done":
            if ev["id"] != req_id:
                fail(f"wrong done id: {ev}")
            return ev, tokens
        elif ev["type"] == "error":
            fail(f"client error: {ev['message']}")


sock_path = os.path.join(tempfile.mkdtemp(), "tlc-smoke.sock")

# 1. listen on a unix socket; stdin stays open (the spawner's client)
proc = subprocess.Popen([runner, "--model", model, "--listen", sock_path, "--token", token],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1)
for _ in range(120):
    if os.path.exists(sock_path):
        break
    if proc.poll() is not None:
        fail(f"runner exited early: {proc.stderr.read()}")
    time.sleep(0.5)
else:
    fail("socket never appeared")

# 2. unix socket client generates without auth while stdin is silent — the
#    openvino pitfall: an unread stdin must not freeze the runner
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
sf = s.makefile("rw")
ev, tokens = generate(sf, 7, "def add(a, b):")
print(f"socket client done: finish={ev['finishReason']} tokens={len(tokens)}")

# 3. generation over stdin while the socket client is connected
proc.stdin.write(json.dumps({"type": "generate", "id": 5, "prompt": "def mul(a, b):",
                             "maxTokens": 16, "stop": ["\n"]}) + "\n")
proc.stdin.flush()
toks = []
while True:
    ev = read_event(proc.stdout)
    if ev["type"] == "token":
        toks.append(ev["text"])
    elif ev["type"] == "done":
        print(f"stdio client done: finish={ev['finishReason']} tokens={len(toks)}")
        break

# 4. taken endpoint: a second runner must exit 1
dup = subprocess.run([runner, "--model", model, "--listen", sock_path],
                     capture_output=True, text=True, timeout=120)
if dup.returncode != 1:
    fail(f"second runner on taken socket: exit={dup.returncode}")
print("taken endpoint rejected: exit=1")

# 5. TCP: ready carries pid+port; token-less TCP client is rejected
tcp = subprocess.Popen([runner, "--model", model, "--listen-tcp", "0", "--token", token],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       text=True, bufsize=1)
ready = read_event(tcp.stdout, "ready")
if not ready.get("pid") or not ready.get("port"):
    fail(f"bad tcp ready: {ready}")
print(f"tcp ready: port={ready['port']} pid={ready['pid']}")
ts = socket.create_connection(("127.0.0.1", ready["port"]))
tf = ts.makefile("rw")
tf.write(json.dumps({"type": "generate", "id": 1, "prompt": "x", "maxTokens": 4, "stop": []}) + "\n")
tf.flush()
err = read_event(tf)
if err["type"] != "error" or "auth" not in err["message"]:
    fail(f"expected auth error, got {err}")
print("token-less TCP client rejected")

# 6. self-exit: both stdin and the socket client go away
s.close()
proc.stdin.close()
if proc.wait(timeout=15) != 0:
    fail(f"runner did not self-exit cleanly: {proc.returncode}")
print("runner self-exit after last client: 0")
tcp.kill()
print("SMOKE OK")
