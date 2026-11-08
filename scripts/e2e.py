#!/usr/bin/env python3
"""Manual client and E2E test suite for the tlc-mlx-runner.

Stdlib-only, macOS arm64 only (same shape as the OpenVINO runner's e2e.py).

Usage:
  python3 scripts/e2e.py test [--model DIR]   # run the test suite
      Without --model only protocol tests that need no weights run.
      With --model DIR (or MODEL_DIR env) the full suite incl. FIM runs.
  python3 scripts/e2e.py manual [--model DIR] # interactive NDJSON client
  python3 scripts/e2e.py download-model [--out DIR]
                                              # fetch the default test model
  python3 scripts/e2e.py bench [--runs N]     # model load time + inference
                                              # speed for the bench models
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")
CACHE_DIR = os.path.join(ROOT, ".cache")

DEFAULT_MODEL_REPO = "mlx-community/Qwen2.5-Coder-0.5B-4bit"
DEFAULT_MODEL_DIR = os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-4bit")

# Bench models: Qwen2.5-Coder (base, not Instruct — the runner speaks raw FIM)
# in mlx-community 4-bit and 8-bit, sizes 0.5B/1.5B/3B/7B — quantization effect
# on load time vs inference speed.
BENCH_MODELS = [
    ("0.5B-4bit", "mlx-community/Qwen2.5-Coder-0.5B-4bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-4bit")),
    ("1.5B-4bit", "mlx-community/Qwen2.5-Coder-1.5B-4bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-4bit")),
    ("3B-4bit", "mlx-community/Qwen2.5-Coder-3B-4bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-4bit")),
    ("0.5B-8bit", "mlx-community/Qwen2.5-Coder-0.5B-8bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-8bit")),
    ("1.5B-8bit", "mlx-community/Qwen2.5-Coder-1.5B-8bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-8bit")),
    ("3B-8bit", "mlx-community/Qwen2.5-Coder-3B-8bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-8bit")),
    ("7B-4bit", "mlx-community/Qwen2.5-Coder-7B-4bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-7b-4bit")),
    ("7B-8bit", "mlx-community/Qwen2.5-Coder-7B-8bit",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-7b-8bit")),
]
BENCH_MAX_TOKENS = 64
BENCH_RUNS = 2

FIM_PROMPT = "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n    return a + b<|fim_middle|>"
FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


def fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def binary_path():
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        fail("the MLX runner is macOS arm64 only")
    p = os.path.join(BUILD_DIR, "tlc-mlx-runner-macos-arm64")
    if not os.path.isfile(p):
        fail(f"{p} not found - run `task -t tasks.yaml build` first")
    return p


class Runner:
    """Spawned runner process; events are collected on a reader thread."""

    def __init__(self, model=None):
        self.binary = binary_path()
        args = [self.binary]
        if model:
            args += ["--model", model]
        self.proc = subprocess.Popen(
            args, cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1)
        self.events = []
        self.lines = []  # raw stdout lines, for non-JSON assertions
        import threading
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.events.append(ev)
            if ev.get("type") == "done":
                text = ""
                for prev in reversed(self.events[:-1]):
                    if prev.get("type") in ("done", "error"):
                        break
                    if prev.get("type") == "token":
                        text = prev.get("text", "") + text
                print(f"<< {ev} text={text!r}", file=sys.stderr)
            elif ev.get("type") == "error":
                print(f"<< {ev}", file=sys.stderr)

    def send(self, obj):
        if obj.get("type") == "generate":
            print(f">> {json.dumps(obj, ensure_ascii=False)}", file=sys.stderr)
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, pred, timeout=120, since=0):
        """Block until an event matching pred appears after index `since`; returns it."""
        deadline = None if timeout is None else time.monotonic() + timeout
        seen = since
        while deadline is None or time.monotonic() < deadline:
            for ev in self.events[seen:]:
                seen += 1
                if pred(ev):
                    return ev
            if self.proc.poll() is not None:
                # the child is gone but the reader thread may not have drained
                # its stdout yet; give it a moment before declaring a timeout
                self._thread.join(2.0)
                for ev in self.events[seen:]:
                    seen += 1
                    if pred(ev):
                        return ev
                break
            time.sleep(0.02)
        raise AssertionError(f"timeout waiting for event (exit={self.proc.poll()}, events={self.events[seen:]})")

    def close(self, timeout=30):
        try:
            self.send({"type": "shutdown"})
        except OSError:
            pass
        try:
            self.proc.stdin.close()  # avoid noisy stdin flush at GC when the child is gone
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.proc.returncode


def check(cond, name):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}")
    return cond


def lastError(r):
    """Message of the first error event (may be empty if the reader thread lost the race)."""
    err = next((e.get("message", "") for e in r.events if e.get("type") == "error"), "")
    return err.strip().replace("\n", " ")[:300] or "(no error event captured)"


# ---------------------------------------------------------------- protocol tests (no model)

def test_no_model():
    print("protocol: no --model given")
    r = Runner()
    code = r.proc.wait(timeout=15)
    ok = check(code == 2, f"exit code is 2 (got {code})")
    ok &= check(any(e.get("type") == "error" and "model" in e.get("message", "") for e in r.events),
                "error event mentions the model")
    return ok


def test_bad_model():
    print("protocol: bad model path")
    r = Runner(model=os.path.join(ROOT, "does-not-exist"))
    code = r.proc.wait(timeout=30)
    ok = check(code == 1, f"exit code is 1 (got {code})")
    ok &= check(any(e.get("type") == "error" and e.get("message", "").startswith("model load failed") for e in r.events),
                "error event starts with 'model load failed'")
    return ok


def test_version():
    print("protocol: --version")
    proc = subprocess.run([binary_path(), "--version"], capture_output=True, text=True, timeout=15)
    ok = check(proc.returncode == 0, f"exit code is 0 (got {proc.returncode})")
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    ok &= check(len(lines) == 1, f"exactly one stdout line (got {len(lines)})")
    try:
        ev = json.loads(lines[0]) if lines else {}
    except json.JSONDecodeError:
        ev = {}
    ok &= check(ev.get("type") == "version" and isinstance(ev.get("app"), str) and ev.get("app") != "unknown",
                f"version event with an app version (got {ev})")
    return ok


def test_unparsable_input(model):
    print("model: unparsable stdin line")
    r = Runner(model=model)
    expect_ready(r, model)
    r.proc.stdin.write("this is not json\n")
    r.proc.stdin.flush()
    ev = r.wait_for(lambda e: e.get("type") == "error")
    ok = check(ev.get("id") is None, "error event has id null")
    r.send({"type": "generate", "id": 99, "prompt": "hi", "maxTokens": 4})
    ok &= check(r.wait_for(lambda e: e.get("type") == "done").get("id") == 99,
                "process survived and completed the next request")
    ok &= check(r.close() == 0, "clean shutdown, exit 0")
    return ok


def test_unknown_type_rejected(model):
    print("model: unknown request type rejected")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "frobnicate"})
    ev = r.wait_for(lambda e: e.get("type") == "error")
    ok = check("unknown request type" in ev.get("message", ""), f"error names the type (got {ev})")
    r.send({"type": "generate", "id": 98, "prompt": "hi", "maxTokens": 4})
    ok &= check(r.wait_for(lambda e: e.get("type") == "done").get("id") == 98, "no crash, request handled")
    ok &= check(r.close() == 0, "clean shutdown, exit 0")
    return ok


def test_stdout_ndjson_only():
    print("protocol: stdout carries JSON lines only")
    r = Runner(model=os.path.join(ROOT, "does-not-exist"))
    r.proc.wait(timeout=30)
    import json as _json
    ok = check(all(_json.loads(l).get("type") for l in r.lines), "every stdout line parses as a JSON object")
    return ok


# ---------------------------------------------------------------- model tests

def expect_ready(r, model):
    ev = r.wait_for(lambda e: e.get("type") == "ready",
                    timeout=int(os.environ.get("E2E_READY_TIMEOUT", "300")))
    return ev


def test_generation(model):
    print("model: streaming generation")
    r = Runner(model=model)
    ready = expect_ready(r, model)
    ok = check(ready.get("device") == "gpu" and isinstance(ready.get("ctx"), int), f"ready event: {ready}")
    ok &= check(all(e.get("type") != "token" or "id" not in e for e in r.events),
                "token events carry no id")
    r.send({"type": "generate", "id": 1, "prompt": "The capital of France is", "maxTokens": 24})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    ok &= check(done.get("id") == 1, "done event with id 1")
    ok &= check(done.get("finishReason") in ("stop", "length"), f"finishReason is stop|length (got {done.get('finishReason')})")
    ok &= check(len(tokens) > 0, f"streamed {len(tokens)} token event(s)")
    ok &= check(all("text" in t for t in tokens), "token events carry text")
    ok &= check("".join(t["text"] for t in tokens).strip() != "", "completion text is non-empty")
    code = r.close()
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


def test_max_tokens(model):
    print("model: maxTokens limit")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 2, "prompt": "tell me a very long story", "maxTokens": 64})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    ok = check(done.get("finishReason") == "length", f"finishReason is length (got {done.get('finishReason')})")
    ok &= check(len(tokens) <= 64, f"at most 64 token events (got {len(tokens)})")
    r.close()
    return ok


def test_stop_strings(model):
    print("model: stop strings")
    r = Runner(model=model)
    expect_ready(r, model)
    # greedy decoding is deterministic: the model counts on
    # ' six seven eight ...' on every supported model, so stopping
    # on 'seven' must cut generation right after ' six '.
    r.send({"type": "generate", "id": 3, "prompt": "count: one two three four five", "maxTokens": 64, "stop": ["seven"]})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    text = "".join(t["text"] for t in tokens)
    ok = check(done.get("finishReason") == "stop", f"finishReason is stop (got {done.get('finishReason')})")
    ok &= check("seven" not in text, f"stop string not present in output: {text!r}")
    ok &= check(len(tokens) < 64, f"generation stopped early ({len(tokens)} tokens)")
    r.close()
    return ok


def test_fim(model):
    print("model: FIM completion")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 4, "prompt": FIM_PROMPT, "maxTokens": 48, "stop": ["\n\n"]})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    text = "".join(t["text"] for t in tokens)
    ok = check(done.get("finishReason") in ("stop", "length"), f"done event (finishReason={done.get('finishReason')})")
    ok &= check(len(tokens) >= 1, f"streamed {len(tokens)} token event(s)")
    ok &= check(len(text.strip()) > 0, f"FIM produced a middle part: {text!r}")
    ok &= check(not any(tok in text for tok in FIM_TOKENS), "no FIM control tokens leak into output")
    r.close()
    return ok


def test_prompt_cache(model):
    print("model: prompt cache reuse")
    r = Runner(model=model)
    expect_ready(r, model)
    body = "\n".join(f"\tlet item{i} = load({i})" for i in range(40))
    prompt = "<|fim_prefix|>func setup() {\n" + body + "\n\tlet done<|fim_suffix|>\n}<|fim_middle|>"
    dones = []
    for run_id in (6, 7):
        r.events.clear()
        r.send({"type": "generate", "id": run_id, "prompt": prompt, "maxTokens": 24, "cache": True})
        dones.append(r.wait_for(lambda e: e.get("type") == "done"))
    ok = check(dones[0].get("cachedTokens") == 0, f"cold run starts from an empty cache (got {dones[0].get('cachedTokens')})")
    expected = dones[0].get("promptTokens", 0) - 1
    ok &= check(dones[1].get("cachedTokens") == expected,
                f"identical prompt reuses all but the last token ({dones[1].get('cachedTokens')} of {expected})")
    ok &= check(dones[1].get("promptTokens") == dones[0].get("promptTokens"), "prompt token count matches across runs")
    code = r.close()
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


def test_cancel(model):
    print("model: cancel mid-generation")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 5, "prompt": "tell me a very long story", "maxTokens": 1024})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120)
    r.send({"type": "cancel", "id": 5})
    done = r.wait_for(lambda e: e.get("type") == "done", timeout=60)
    ok = check(done.get("finishReason") == "cancel", f"finishReason is cancel (got {done.get('finishReason')})")
    ok &= check(done.get("id") == 5, "done event carries the request id")
    code = r.close()
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


def test_queue(model):
    print("model: request queue (no busy, cancel by id, shutdown drain)")
    r = Runner(model=model)
    expect_ready(r, model)
    # two generates back to back: the runner queues them, no busy rejection
    r.send({"type": "generate", "id": 6, "prompt": "tell me a story", "maxTokens": 64})
    r.send({"type": "generate", "id": 7, "prompt": "another"})
    ok = check(r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 6, timeout=300).get("finishReason") in ("stop", "length"),
               "id 6 completed first (queue order)")
    ok &= check(r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 7, timeout=300) is not None,
                "id 7 completed after id 6 (queued, no busy error)")
    ok &= check(not any(e.get("type") == "error" for e in r.events), "no error events (busy removed)")
    # cancel a queued request: resolved immediately with finishReason=cancel
    r.send({"type": "generate", "id": 8, "prompt": "tell me a very long story", "maxTokens": 1024})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120)
    r.send({"type": "generate", "id": 9, "prompt": "queued"})
    r.send({"type": "cancel", "id": 9})
    ev = r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 9, timeout=30)
    ok &= check(ev.get("finishReason") == "cancel", f"queued id 9 cancelled (got {ev})")
    # cancel the active generation
    r.send({"type": "cancel", "id": 8})
    ev = r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 8, timeout=60)
    ok &= check(ev.get("finishReason") == "cancel", f"active id 8 cancelled (got {ev})")
    # shutdown drains queued requests with finishReason=cancel
    r.send({"type": "generate", "id": 10, "prompt": "tell me a very long story", "maxTokens": 1024})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120)
    r.send({"type": "generate", "id": 11, "prompt": "queued"})
    r.send({"type": "shutdown"})
    ev = r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 11, timeout=30)
    ok &= check(ev.get("finishReason") == "cancel", f"shutdown drained id 11 with cancel (got {ev})")
    ok &= check(r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 10, timeout=60).get("finishReason") == "cancel",
                "shutdown cancelled the active id 10")
    code = r.proc.wait(timeout=60)
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


# ---------------------------------------------------------------- bench

def bench_generate(r, run_id, prompt, use_cache=False):
    """One fixed-length FIM generation; returns (ttft_s, total_s, tokens, promptTokens, cachedTokens)."""
    start = len(r.events)
    t0 = time.monotonic()
    payload = {"type": "generate", "id": run_id, "prompt": prompt, "maxTokens": BENCH_MAX_TOKENS}
    if use_cache:
        payload["cache"] = True
    r.send(payload)
    r.wait_for(lambda e: e.get("type") == "token", timeout=120, since=start)
    ttft = time.monotonic() - t0
    done = r.wait_for(lambda e: e.get("type") == "done", timeout=300, since=start)
    total = time.monotonic() - t0
    tokens = sum(1 for e in r.events[start:] if e.get("type") == "token")
    if done.get("finishReason") not in ("length", "stop"):
        raise AssertionError(f"bench run {run_id} failed: {done}")
    return ttft, total, tokens, done.get("promptTokens"), done.get("cachedTokens")


def _sys_info():
    """Host characteristics for the bench header: computer model, CPU, RAM, OS."""
    info = {}

    def first_line(*args):
        try:
            return subprocess.check_output(args, text=True).strip()
        except Exception:
            return None

    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Model Name:"):
                    info["host"] = line.split(":", 1)[1].strip()
                elif line.startswith("Chip:"):
                    info["cpu"] = line.split(":", 1)[1].strip()
        except Exception:
            pass
        ram = first_line("sysctl", "-n", "hw.memsize")
        if ram and ram.isdigit():
            info["ram"] = f"{int(ram) // (1 << 30)} GB"
        ver = first_line("sw_vers", "-productVersion")
        if ver:
            info["os"] = f"macOS {ver}"
    elif os.name == "nt":
        out = first_line("powershell", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_ComputerSystem).Model")
        if out:
            info["host"] = out
        out = first_line("powershell", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_Processor).Name")
        if out:
            info["cpu"] = out.splitlines()[0].strip()
        out = first_line("powershell", "-NoProfile", "-Command",
                         "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)")
        if out and out.isdigit():
            info["ram"] = f"{out} GB"
        out = first_line("powershell", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_OperatingSystem).Caption")
        if out:
            info["os"] = out
    else:
        try:
            with open("/sys/devices/virtual/dmi/id/product_name") as f:
                info["host"] = f.read().strip()
        except Exception:
            pass
        out = first_line("sh", "-c", "lscpu | sed -n 's/^Model name:\\s*//p'")
        if out:
            info["cpu"] = out.splitlines()[0].strip()
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram"] = f"{int(line.split()[1]) // (1 << 20)} GB"
                        break
        except Exception:
            pass
        out = first_line("sh", "-c", ". /etc/os-release && echo $PRETTY_NAME")
        if out:
            info["os"] = out
    return info


def bench(runs=BENCH_RUNS, sizes=None, prompt_kind="fixture", cache=False):
    """Download the bench models, measure model load time and inference speed on
    each. ttft_s carries prefill; gen_s/tok_s are decode-only. With cache=True
    each model also runs warm continuations (typed characters at the cursor) so
    ttft_w shows the KV-cache reuse gain and cached_tok the reuse reported by
    the runner."""
    prompt, prefix_chars, suffix_chars = fixture_prompt_parts() if prompt_kind == "fixture" else (FIM_PROMPT, len(FIM_PROMPT), 0)
    warm_prompts = []
    if cache:
        if prompt_kind == "fixture":
            warm_prompts = [fixture_prompt_parts(typed=t)[0] for t in ("a", "ab")]
        else:
            warm_prompts = [prompt, prompt]
    rows = []
    warm_rows = []
    prompt_tokens = None
    ready_ev = None
    for name, repo, out in BENCH_MODELS:
        if sizes and name not in sizes:
            continue
        out = os.environ.get(f"E2E_{name.replace('.', '_').replace('-', '_')}_DIR", out)
        if not os.path.isfile(os.path.join(out, "config.json")):
            print(f"bench: downloading {name} ({repo})", flush=True)
            download_model(out, repo)
        print(f"bench: {name}", flush=True)
        gens, warm = [], []
        for attempt in range(2):
            try:
                r = Runner(model=out)
                t0 = time.monotonic()
                ready_ev = r.wait_for(lambda e: e.get("type") == "ready", timeout=None)
                load_s = time.monotonic() - t0
                gens = []
                for i in range(runs):
                    gens.append(bench_generate(r, i, prompt))
                warm = []
                for j, wp in enumerate(warm_prompts):
                    warm.append(bench_generate(r, 100 + j, wp, use_cache=True))
                r.close()
                break
            except AssertionError:
                try:
                    r.close()
                except Exception:
                    pass
                gens, warm = [], []
                if attempt == 1:
                    raise
                print(f"  {name}: no token within 120s (rare MLX stall), retrying", flush=True)
        steady = gens[0] if cache else gens[-1]  # cache mode: runs 2+ would ride the warm cache
        ttft, total, tokens, prompt_tokens, _ = steady
        decode = max(total - ttft, 1e-3)
        rows.append((name, load_s, ttft, decode, tokens))
        print(f"  {name}: load={load_s:.1f}s ttft={ttft:.2f}s "
              f"decode={decode:.2f}s ({tokens} tok, {tokens/decode:.1f} tok/s)", flush=True)
        if warm:
            wttft, _, _, wptok, wcached = warm[-1]  # steady warm state
            warm_rows.append((name, wttft, wcached, wptok))
            print(f"  {name}: warm ttft={wttft:.2f}s cached={wcached}/{wptok}", flush=True)
    # Print device characteristics before the final table
    sysinfo = _sys_info()
    ctx = ready_ev.get("ctx", "?") if ready_ev else "?"
    print(f"\n== gpu ==  host={sysinfo.get('host', '?')}  cpu={sysinfo.get('cpu', '?')}  "
          f"ram={sysinfo.get('ram', '?')}  os={sysinfo.get('os', '?')}  ctx={ctx}  "
          f"runs={runs}  maxTokens={BENCH_MAX_TOKENS}  "
          f"prompt={len(prompt)} chars = {prefix_chars} prefix + {suffix_chars} suffix, "
          f"{prompt_tokens if prompt_tokens is not None else '?'} tokens")
    w = max((len(name) for name, *_rest in rows), default=len("model"))
    print(f"{'model':<{w}}  load_s  ttft_s  gen_s  tokens  tok_s")
    for name, l, ttft, total, tokens in rows:
        print(f"{name:<{w}} {l:>7.1f} {ttft:>7.2f} {total:>6.2f} {tokens:>6} {tokens/total:>7.1f}")
    if warm_rows:
        print(f"\nwarm start (KV-cache reuse, prompt + typed characters):")
        print(f"{'model':<{w}}  ttft_w_s  cached_tok")
        for name, wttft, wcached, wptok in warm_rows:
            print(f"{name:<{w}} {wttft:>9.2f}  {wcached if wcached is not None else '?'}/{wptok}")


# ---------------------------------------------------------------- model download

def hf_api(repo):
    out = subprocess.run(["curl", "-fsSL", f"https://huggingface.co/api/models/{repo}"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fetch(url, dest):
    """Resumable download via curl; tolerates flaky connections."""
    tmp = dest + ".part"
    if os.path.isfile(dest):
        print(f"  exists  {os.path.basename(dest)}")
        return
    print(f"  fetch   {os.path.basename(dest)}")
    for attempt in range(1, 11):
        r = subprocess.run(["curl", "-fSsL", "-C", "-", "--retry", "3", "--retry-delay", "2", url, "-o", tmp])
        if r.returncode == 0:
            os.replace(tmp, dest)
            return
        print(f"  attempt {attempt} failed (curl {r.returncode}), resuming...", file=sys.stderr)
        time.sleep(3)
    subprocess.run(["curl", "-fSsL", "-C", "-", url, "-o", tmp], check=True)  # last try, raise
    os.replace(tmp, dest)


def download_model(out_dir, repo=DEFAULT_MODEL_REPO):
    info = hf_api(repo)
    files = [s["rfilename"] for s in info.get("siblings", [])]
    keep = [f for f in files if f in ("config.json", "generation_config.json", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                                      "model.safetensors.index.json")
            or f.endswith(".safetensors")]
    if not any(f.endswith(".safetensors") for f in keep):
        print(f"warning: no safetensors weights listed in {repo}", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)
    for f in keep:
        dest = os.path.join(out_dir, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fetch(f"https://huggingface.co/{repo}/resolve/main/{f}", dest)


# ---------------------------------------------------------------- manual client

# ---------------------------------------------------------------- real file FIM

# The plugin's qwen2 template stop list and context clamps (src/templates/builtins.ts).
QWEN2_RESERVED = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>",
                  "<|repo_name|>", "<|file_sep|>", "<|end_of_text|>", "<|endoftext|>"]
PLUGIN_PREFIX_MAX = 8000
PLUGIN_SUFFIX_MAX = 2000


def mask_reserved(text):
    for token in QWEN2_RESERVED:
        text = text.replace(token, token[0] + " " + token[1:])
    return text


def fixture_prompt_parts(typed=""):
    """Plugin-shaped FIM prompt over the bundled fixture: the cursor sits on the
    whitespace-only line 96 inside a var block, and the prefix/suffix clamps
    mirror the plugin defaults. `typed` appends characters at the cursor the
    way typing does (prefix grows, suffix unchanged) for cache-warm runs; the
    prefix window is not re-clamped, so the front stays anchored like in files
    that fit the prefix budget. Returns (prompt, prefix_chars, suffix_chars)."""
    with open(os.path.join(ROOT, "scripts", "fixtures", "api_test.go"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    li, col = 95, 2
    prefix = mask_reserved("\n".join(lines[:li]) + "\n" + lines[li][:col])[-PLUGIN_PREFIX_MAX:] + typed
    suffix = mask_reserved(lines[li][col:] + "\n" + "\n".join(lines[li + 1:]))[:PLUGIN_SUFFIX_MAX]
    return ("<|fim_prefix|>" + prefix + "<|fim_suffix|>" + suffix + "<|fim_middle|>",
            len(prefix), len(suffix))


def fixture_fim_prompt():
    return fixture_prompt_parts()[0]


def real_file(model, file, line, column, max_tokens):
    """Raw FIM against a real source file at a 1-based line/column cursor.

    Mirrors the plugin's prompt (qwen2 markers, 8000/2000 clamp, reserved-token
    masking) but does NOT send a newline stop — the point is to observe the
    backend's natural (possibly multi-line) completion.
    """
    with open(file, encoding="utf-8") as f:
        lines = f.read().split("\n")
    li = max(0, min(line - 1, len(lines) - 1))
    col = min(max(1, column) - 1, len(lines[li]))
    prefix = mask_reserved("\n".join(lines[:li]) + "\n" + lines[li][:col])[-PLUGIN_PREFIX_MAX:]
    suffix = mask_reserved(lines[li][col:] + "\n" + "\n".join(lines[li + 1:]))[:PLUGIN_SUFFIX_MAX]
    prompt = "<|fim_prefix|>" + prefix + "<|fim_suffix|>" + suffix + "<|fim_middle|>"
    next_line = lines[li + 1] if li + 1 < len(lines) else ""

    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 1, "prompt": prompt, "maxTokens": max_tokens, "stop": QWEN2_RESERVED})
    done = r.wait_for(lambda e: e.get("type") == "done", timeout=300)
    tokens = [e for e in r.events if e.get("type") == "token"]
    text = "".join(t["text"] for t in tokens)
    r.close()

    completion_lines = text.split("\n")
    print("\n" + "=" * 72)
    print(f"file: {file} @ {line}:{column} (cursor line: {lines[li][:col]!r})")
    print(f"next line in file: {next_line!r}")
    print(f"completion: {len(text)} chars, {len(completion_lines)} lines, "
          f"finishReason={done.get('finishReason') if done else 'TIMEOUT'}, "
          f"prompt={done.get('promptTokens') if done else '?'} tokens")
    stripped = text.strip()
    next_trim = next_line.strip()
    if not stripped:
        print("overlap: empty completion")
    elif stripped == next_trim:
        print("overlap: completion duplicates the next line exactly")
    elif next_trim and stripped.startswith(next_trim):
        print(f"overlap: completion extends the next line by {stripped[len(next_trim):]!r}")
    else:
        common = os.path.commonprefix([stripped, next_trim])
        share = round(100 * len(common) / max(1, len(stripped)))
        print(f"overlap: common prefix with the next line is {len(common)} chars (~{share}% of the completion): {common!r}")
    print("-" * 72)
    print(text)
    print("=" * 72)
    ok = check(done is not None, "done event received")
    ok &= check(not any(tok in text for tok in QWEN2_RESERVED), "no FIM control tokens leak into output")
    return ok


def manual_client(model):
    if not model:
        fail("manual mode needs --model or MODEL_DIR")
    print("interactive client — type a JSON request line, or /fim <code>, /quit")
    r = Runner(model=model)
    ready = r.wait_for(lambda e: e.get("type") == "ready", timeout=300)
    print(f"ready: model={ready.get('model')} device={ready.get('device')} ctx={ready.get('ctx')}")

    import threading

    def printer():
        idx = 0
        while True:
            events = r.events
            while idx < len(events):
                ev = events[idx]
                idx += 1
                t = ev.get("type")
                if t == "token":
                    sys.stdout.write(ev.get("text", ""))
                    sys.stdout.flush()
                elif t == "done":
                    print(f"\n[done id={ev.get('id')} finishReason={ev.get('finishReason')}]")
                elif t == "error":
                    print(f"\n[error id={ev.get('id')} message={ev.get('message')}]")
                elif t == "ready":
                    pass
                else:
                    print(f"\n[{json.dumps(ev)}]")

    threading.Thread(target=printer, daemon=True).start()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "/quit":
                break
            if line.startswith("/fim"):
                code = line[4:]
                prompt = f"<|fim_prefix|>{code}<|fim_suffix|><|fim_middle|>"
                r.send({"type": "generate", "id": int(time.time()), "prompt": prompt, "maxTokens": 64})
                continue
            try:
                r.send(json.loads(line))
            except json.JSONDecodeError:
                print("not JSON — ignored")
    except KeyboardInterrupt:
        pass
    code = r.close()
    print(f"runner exited with {code}")


# ---------------------------------------------------------------- main

PROTOCOL_TESTS = [test_no_model, test_bad_model, test_version, test_stdout_ndjson_only]
MODEL_TESTS = [test_unparsable_input, test_unknown_type_rejected, test_generation,
               test_max_tokens, test_stop_strings, test_fim, test_prompt_cache,
               test_cancel, test_queue]


def run_suite(model):
    ok = True
    print(f"== protocol tests (no model) ==")
    for t in PROTOCOL_TESTS:
        ok &= t()
    if model:
        if not (os.path.isfile(os.path.join(model, "config.json")) and
                any(f.endswith(".safetensors") for f in os.listdir(model))):
            fail(f"{model} does not look like an mlx-community model directory")
        print(f"== model tests ({model}) ==")
        for t in MODEL_TESTS:
            ok &= t(model)
    else:
        print("== model tests skipped (no --model; pass --model or set MODEL_DIR) ==")
    print(f"\n{'ALL PASSED' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cmd = ap.add_subparsers(dest="cmd", required=True)
    for name in ("test", "manual"):
        p = cmd.add_parser(name)
        p.add_argument("--model", default=os.environ.get("MODEL_DIR"), help="mlx-community model directory")
    p = cmd.add_parser("real-file")
    p.add_argument("--file", required=True, help="source file to complete")
    p.add_argument("--line", type=int, default=1, help="1-based cursor line")
    p.add_argument("--column", type=int, default=1, help="1-based cursor column")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--model", default=os.environ.get("MODEL_DIR"), help="mlx-community model directory")
    p = cmd.add_parser("download-model")
    p.add_argument("--repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--out", default=DEFAULT_MODEL_DIR)
    p = cmd.add_parser("bench")
    p.add_argument("--runs", type=int, default=BENCH_RUNS, help="generation runs per model (steady state is reported)")
    p.add_argument("--prompt", choices=("fixture", "tiny"), default="fixture",
                   help="fixture = plugin-shaped FIM from the bundled fixture (~1.5–2k token prefill); tiny = one-line FIM")
    p.add_argument("--sizes", default="0.5B-4bit,1.5B-4bit,3B-4bit,7B-4bit,0.5B-8bit,1.5B-8bit,3B-8bit,7B-8bit",
                   help="comma-separated model sizes to bench (model dirs: E2E_0_5B_4bit_DIR / E2E_1_5B_4bit_DIR / E2E_3B_4bit_DIR / E2E_7B_4bit_DIR / E2E_0_5B_8bit_DIR / E2E_1_5B_8bit_DIR / E2E_3B_8bit_DIR / E2E_7B_8bit_DIR)")
    p.add_argument("--cache", action="store_true",
                   help="also run warm continuations (KV-cache reuse) and report ttft_w/cached_tok")
    args = ap.parse_args()

    if args.cmd == "download-model":
        download_model(args.out, args.repo)
        print(f"model at {args.out}")
    elif args.cmd == "bench":
        bench(args.runs, args.sizes.split(","), args.prompt, cache=args.cache)
    elif args.cmd == "manual":
        manual_client(args.model)
    elif args.cmd == "real-file":
        sys.exit(0 if real_file(args.model, args.file, args.line, args.column, args.max_tokens) else 1)
    else:
        run_suite(args.model)


if __name__ == "__main__":
    main()
