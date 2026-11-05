"""Chaos jobs: a fake training/data job that fails with a chosen fault. Standard library only.

    python3 -m lab.chaos --fault oom.memoryerror --seed 3 --style hf --fail-attempts 1

Faults are real wherever a CPU box can produce them (MemoryError under an address-space
limit, ENOSPC from /dev/full, refused and timed-out sockets, DNS failures, HTTP 503 from a
local server, held file locks, corrupt JSON, real Python bugs). GPU faults cannot happen
on a CPU, so those variants raise exceptions whose type and message are copied from the
library that would raise them (`emulated` in FAULTS). Every fault runs inside code that is
compiled under a realistic site-packages filename and function name, so tracebacks look
like the real thing and never mention the label.

The job behaves like a resumable training job, which is what lets the healer prove a fix:

* it writes `checkpoint-N/` folders under `--run-dir` and resumes from `LAB_RESUME_FROM`;
* `oom.*` faults stop once `LAB_BATCH_SCALE` <= `--oom-below` (the healer halves it);
* `disk_full.*` faults stop once `--run-dir` holds 2 or fewer checkpoints (the healer prunes);
* other faults fire only while `LAB_ATTEMPT` <= `--fail-attempts`;
* `bad_data.*` and `code_bug.*` fire on every attempt, as they would in real life.
"""

from __future__ import annotations

import argparse
import contextlib
import linecache
import os
import random
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SITE = "/home/lab/.venv/lib/python3.12/site-packages"


@dataclass(frozen=True)
class Fault:
    label: str
    real: bool
    file: str      # filename shown in the traceback
    src: str       # defines `def run(ctx)` plus helpers; the traceback shows these frames


FAULTS: dict[str, Fault] = {}


def fault(name: str, label: str, real: bool, file: str, src: str) -> None:
    FAULTS[name] = Fault(label, real, file, src)


# ---- oom ---------------------------------------------------------------------------------
fault("oom.memoryerror", "oom", True, "train/collate.py", '''
import resource
def pad_batch(rows, width):
    buf = bytearray(len(rows) * width * 1024 * 1024)
    return buf
def run(ctx):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    with open("/proc/self/statm") as fh:
        vsz = int(fh.read().split()[0]) * resource.getpagesize()
    resource.setrlimit(resource.RLIMIT_AS, (vsz + (256 << 20), hard))
    rows = list(range(ctx["batch"]))
    return pad_batch(rows, 4096)
''')
fault("oom.torch_cuda", "oom", False, "torch/nn/modules/linear.py", '''
class OutOfMemoryError(RuntimeError):
    pass
OutOfMemoryError.__module__ = "torch.cuda"
OutOfMemoryError.__qualname__ = "OutOfMemoryError"
def forward(self_, input):
    gib = ctx_["alloc"]
    raise OutOfMemoryError(
        f"CUDA out of memory. Tried to allocate {gib:.2f} GiB. GPU 0 has a total capacity of 119.70 GiB "
        f"of which {ctx_['free']:.2f} GiB is free. Including non-PyTorch memory, this process has "
        f"{119.70 - ctx_['free']:.2f} GiB memory in use. Of the allocated memory {ctx_['used']:.2f} GiB is "
        "allocated by PyTorch, and 312.45 MiB is reserved by PyTorch but unallocated. If reserved but "
        "unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid "
        "fragmentation.  See documentation for Memory Management  "
        "(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)")
def run(ctx):
    global ctx_
    ctx_ = ctx
    return forward(None, None)
''')
fault("oom.numpy_alloc", "oom", False, "numpy/_core/numeric.py", '''
class _ArrayMemoryError(MemoryError):
    pass
_ArrayMemoryError.__module__ = "numpy._core._exceptions"
def zeros(shape, dtype="float32"):
    n = 1
    for s in shape:
        n *= s
    raise _ArrayMemoryError(f"Unable to allocate {n * 4 / 2**30:.1f} GiB for an array with shape {shape} "
                            f"and data type {dtype}")
def run(ctx):
    return zeros((ctx["batch"] * 1024, 65536, 8))
''')
fault("oom.vllm_kv", "oom", False, "vllm/v1/core/kv_cache_utils.py", '''
def check_enough_kv_cache_memory(cfg, available):
    if available <= 0:
        raise ValueError("No available memory for the cache blocks. Try increasing `gpu_memory_utilization` "
                         "when initializing the engine.")
def run(ctx):
    print("INFO [gpu_worker.py:276] Available KV cache memory: -1.85 GiB")
    check_enough_kv_cache_memory(None, -1.85)
''')
fault("oom.sigkill", "oom", True, "train/loop.py", '''
import os, signal
def run(ctx):
    print("Killed")
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)
''')
fault("oom.cublas_alloc", "oom", False, "torch/nn/functional.py", '''
def linear(x, w):
    raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`")
def run(ctx):
    return linear(None, None)
''')

# ---- disk full ---------------------------------------------------------------------------
fault("disk_full.devfull", "disk_full", True, "transformers/trainer.py", '''
def _save_checkpoint(model, out):
    with open("/dev/full", "wb") as fh:
        fh.write(b"\\0" * (8 << 20))
        fh.flush()
def run(ctx):
    print(f"Saving model checkpoint to {ctx['run_dir']}/checkpoint-{ctx['step']}")
    _save_checkpoint(None, ctx["run_dir"])
''')
fault("disk_full.torch_save", "disk_full", False, "torch/serialization.py", '''
def _save(obj, zip_file):
    raise RuntimeError("[enforce fail at inline_container.cc:603] . unexpected pos 2340563584 vs 2340563472")
def save(obj, f):
    try:
        _save(obj, f)
    except RuntimeError:
        raise RuntimeError("PytorchStreamWriter failed writing file data/71: file write failed")
def run(ctx):
    save({}, ctx["run_dir"])
''')
fault("disk_full.quota", "disk_full", True, "datasets/arrow_writer.py", '''
import errno
def write_batch(path):
    raise OSError(errno.EDQUOT, os.strerror(errno.EDQUOT), path)
def run(ctx):
    write_batch(ctx["run_dir"] + "/cache-00017.arrow")
''')
fault("disk_full.hf_space", "disk_full", False, "datasets/builder.py", '''
def _check_disk(needed, free):
    raise OSError(f"Not enough disk space. Needed: {needed:.2f} GiB (download: 4.12 GiB, generated: "
                  f"{needed - 4.12:.2f} GiB, post-processed: Unknown size)")
def download_and_prepare(ctx):
    _check_disk(ctx["needed"], 0.4)
def run(ctx):
    download_and_prepare(ctx)
''')
fault("disk_full.sqlite", "disk_full", False, "lab_tools/cache.py", '''
import sqlite3
def put(key, value):
    raise sqlite3.OperationalError("database or disk is full")
def run(ctx):
    put("row-91233", b"x")
''')

# ---- cuda error (emulated: no GPU here) --------------------------------------------------
_CUDA_TAIL = ("\\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace "
              "below might be incorrect.\\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\\n"
              "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\\n")
fault("cuda_error.illegal_access", "cuda_error", False, "torch/_tensor.py", f'''
class AcceleratorError(RuntimeError):
    pass
AcceleratorError.__module__ = "torch"
def backward(self_):
    raise AcceleratorError("CUDA error: an illegal memory access was encountered{_CUDA_TAIL}")
def run(ctx):
    backward(None)
''')
fault("cuda_error.launch_failure", "cuda_error", False, "torch/cuda/__init__.py", f'''
def synchronize(device=None):
    raise RuntimeError("CUDA error: unspecified launch failure{_CUDA_TAIL}")
def run(ctx):
    synchronize()
''')
fault("cuda_error.cudnn", "cuda_error", False, "torch/nn/modules/conv.py", '''
def _conv_forward(x, w, b):
    raise RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
def run(ctx):
    _conv_forward(None, None, None)
''')
fault("cuda_error.nccl", "cuda_error", False, "torch/distributed/distributed_c10d.py", '''
class DistBackendError(RuntimeError):
    pass
DistBackendError.__module__ = "torch.distributed"
def all_reduce(t):
    raise DistBackendError("NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970, unhandled "
                           "cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5\\n"
                           "ncclUnhandledCudaError: Call to CUDA function failed.\\n"
                           "Last error:\\nCuda failure 'unspecified launch failure'")
def run(ctx):
    all_reduce(None)
''')
fault("cuda_error.xid", "cuda_error", False, "torch/cuda/__init__.py", '''
def _lazy_init():
    raise RuntimeError("CUDA driver error: unknown error")
def run(ctx):
    print("NVRM: Xid (PCI:000f:01:00): 79, pid=48213, GPU has fallen off the bus.")
    _lazy_init()
''')
fault("cuda_error.cublas_exec", "cuda_error", False, "torch/nn/functional.py", '''
def scaled_dot_product_attention(q, k, v):
    raise RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmStridedBatchedEx( "
                       "handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, stridea, b, CUDA_R_16BF, ldb, "
                       "strideb, &fbeta, c, CUDA_R_16BF, ldc, stridec, num_batches, compute_type, "
                       "CUBLAS_GEMM_DEFAULT_TENSOR_OP)`")
def run(ctx):
    scaled_dot_product_attention(None, None, None)
''')

# ---- network -----------------------------------------------------------------------------
fault("network.refused", "network", True, "openai_compat/client.py", '''
import urllib.request
def _send_request(url, body):
    return urllib.request.urlopen(url, data=body, timeout=2)
def chat(ctx):
    return _send_request(f"http://127.0.0.1:{ctx['closed_port']}/v1/chat/completions", b"{}")
def run(ctx):
    chat(ctx)
''')
fault("network.dns", "network", True, "huggingface_hub/file_download.py", '''
import socket
def _get_metadata(host):
    return socket.create_connection((host, 443), timeout=2)
def hf_hub_download(repo_id):
    return _get_metadata("hub-mirror.lab.invalid")
def run(ctx):
    hf_hub_download("google/gemma-4-31b-it")
''')
fault("network.timeout", "network", True, "lab_tools/teacher.py", '''
import socket
def _read_response(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=0.3)
    s.sendall(b"POST /v1/completions HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n")
    return s.recv(1024)
def generate(ctx):
    return _read_response(ctx["silent_port"])
def run(ctx):
    generate(ctx)
''')
fault("network.http503", "network", True, "lab_tools/teacher.py", '''
import urllib.request
def _post(url):
    return urllib.request.urlopen(url, data=b"{}", timeout=2)
def generate(ctx):
    return _post(f"http://127.0.0.1:{ctx['http503_port']}/v1/chat/completions")
def run(ctx):
    generate(ctx)
''')
fault("network.requests_retries", "network", False, "requests/adapters.py", '''
class ConnectionError(OSError):
    pass
ConnectionError.__module__ = "requests.exceptions"
def send(request):
    raise ConnectionError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: "
                          "/api/datasets/JehanzebAli11/scam-radar-sms/revision/main (Caused by "
                          "NameResolutionError(\\"<urllib3.connection.HTTPSConnection object at 0xffff8c2b1d90>: "
                          "Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)\\"))")
def run(ctx):
    send(None)
''')
fault("network.reset", "network", False, "httpx/_transports/default.py", '''
class RemoteProtocolError(Exception):
    pass
RemoteProtocolError.__module__ = "httpx"
def handle_request(req):
    try:
        raise ConnectionResetError(104, "Connection reset by peer")
    except ConnectionResetError:
        raise RemoteProtocolError("Server disconnected without sending a response.")
def run(ctx):
    handle_request(None)
''')

# ---- bad data ----------------------------------------------------------------------------
fault("bad_data.jsonl", "bad_data", True, "lab_tools/io.py", '''
import json
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows
def run(ctx):
    p = ctx["tmp"] + "/train-00003.jsonl"
    with open(p, "w") as fh:
        fh.write('{"text": "ok", "label": 1}\\n' * 3 + '{"text": "Assalam o alaikum, your parcel is held at cust')
    read_jsonl(p)
''')
fault("bad_data.unicode", "bad_data", True, "lab_tools/io.py", '''
def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()
def run(ctx):
    p = ctx["tmp"] + "/sms_export.csv"
    with open(p, "wb") as fh:
        fh.write(b"id,text,label\\n1,caf\\xe9 offer \\xff\\xfe win,1\\n")
    read_text(p)
''')
fault("bad_data.missing_column", "bad_data", True, "datasets/formatting.py", '''
def format_row(row):
    return {"input_ids": row["text"], "labels": row["label"]}
def map_rows(rows):
    return [format_row(r) for r in rows]
def run(ctx):
    rows = [{"text": "a", "label": 0}] * 5 + [{"text": "b", "lable": 1}]
    map_rows(rows)
''')
fault("bad_data.csv_float", "bad_data", True, "lab_tools/features.py", '''
import csv, io
def parse(fh):
    return [(float(r[0]), float(r[1])) for r in csv.reader(fh)]
def run(ctx):
    parse(io.StringIO("0.12,0.5\\n0.33,0.9\\n0.71,N/A?\\n"))
''')
fault("bad_data.gzip", "bad_data", True, "lab_tools/io.py", '''
import gzip
def open_shard(path):
    with gzip.open(path, "rt") as fh:
        return fh.read()
def run(ctx):
    p = ctx["tmp"] + "/shard-0007.jsonl.gz"
    with open(p, "wb") as fh:
        fh.write(b"\\x1f\\x8b\\x08\\x00garbage-after-a-copy-that-was-cut-off")
    open_shard(p)
''')
fault("bad_data.empty", "bad_data", True, "train/data.py", '''
def build_dataset(rows, min_len):
    kept = [r for r in rows if len(r) >= min_len]
    if not kept:
        raise ValueError(f"dataset is empty after filtering ({len(rows)} rows, min_len={min_len})")
    return kept
def run(ctx):
    build_dataset(["", "", ""], 8)
''')
fault("bad_data.label_assert", "bad_data", False, "torch/nn/functional.py", f'''
def cross_entropy(logits, target):
    print("../aten/src/ATen/native/cuda/Loss.cu:250: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], "
          "thread: [5,0,0] Assertion `t >= 0 && t < n_classes` failed.")
    raise RuntimeError("CUDA error: device-side assert triggered{_CUDA_TAIL}")
def run(ctx):
    cross_entropy(None, None)
''')
fault("bad_data.pickle", "bad_data", True, "lab_tools/cache.py", '''
import pickle
def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
def run(ctx):
    p = ctx["tmp"] + "/features.pkl"
    with open(p, "wb") as fh:
        fh.write(b"<html>404 Not Found</html>")
    load(p)
''')

# ---- flaky (succeeds on retry) -----------------------------------------------------------
fault("flaky.flock", "flaky", True, "lab_tools/locks.py", '''
import fcntl
def acquire(path):
    fh = open(path, "w")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fh
def run(ctx):
    p = ctx["tmp"] + "/.hf_cache.lock"
    holder = acquire(p)
    acquire(p)
''')
fault("flaky.port", "flaky", True, "lab_tools/serve.py", '''
import socket
def bind(port):
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    s.listen()
    return s
def run(ctx):
    a = bind(0)
    bind(a.getsockname()[1])
''')
fault("flaky.sqlite_locked", "flaky", True, "lab_tools/cache.py", '''
import sqlite3
def put(db, key):
    con = sqlite3.connect(db, timeout=0)
    con.execute("insert into kv values (?)", (key,))
    con.commit()
def run(ctx):
    db = ctx["tmp"] + "/cache.db"
    a = sqlite3.connect(db)
    a.execute("create table if not exists kv (k text)")
    a.commit()
    a.execute("begin exclusive")
    put(db, "row-1")
''')
fault("flaky.txtbsy", "flaky", True, "lab_tools/exec.py", '''
import subprocess
def run_tool(path):
    return subprocess.run([path], check=True)
def run(ctx):
    p = ctx["tmp"] + "/tokenize.sh"
    fh = open(p, "w")
    fh.write("#!/bin/sh\\necho hi\\n")
    fh.flush()
    os.chmod(p, 0o755)
    run_tool(p)
''')
fault("flaky.stale", "flaky", False, "lab_tools/io.py", '''
import errno
def stat_shard(path):
    raise OSError(errno.ESTALE, os.strerror(errno.ESTALE), path)
def run(ctx):
    stat_shard("/mnt/nas/lab/data/shard-0003.arrow")
''')
fault("flaky.eintr", "flaky", False, "multiprocessing/connection.py", '''
import errno
def _recv(size):
    raise InterruptedError(errno.EINTR, "Interrupted system call")
def recv_bytes():
    return _recv(4)
def run(ctx):
    print("DataLoader worker (pid 51234) is restarting")
    recv_bytes()
''')

# ---- code bugs ---------------------------------------------------------------------------
fault("code_bug.type", "code_bug", True, "train/loop.py", '''
def lr_at(step, cfg):
    return cfg["lr"] * min(1.0, step / cfg["warmup"])
def run(ctx):
    lr_at(10, {"lr": "3e-4", "warmup": 100})
''')
fault("code_bug.attribute", "code_bug", True, "train/model.py", '''
def build_head(cfg):
    head = cfg.get("head")
    return head.num_labels
def run(ctx):
    build_head({})
''')
fault("code_bug.name", "code_bug", True, "train/eval.py", '''
def evaluate(preds, gold):
    correct = sum(p == g for p, g in zip(preds, gold))
    return correct / len(golds)
def run(ctx):
    evaluate([1, 0], [1, 1])
''')
fault("code_bug.index", "code_bug", True, "train/data.py", '''
def collate(batch):
    return [batch[i] for i in range(len(batch) + 1)]
def run(ctx):
    collate([1, 2, 3])
''')
fault("code_bug.shape", "code_bug", True, "train/model.py", '''
def check_shapes(hidden, expected):
    assert hidden == expected, f"hidden size {hidden} != {expected}"
def run(ctx):
    check_shapes(4096, 5120)
''')
fault("code_bug.config_key", "code_bug", True, "train/config.py", '''
def resolve(cfg):
    return cfg["optimizer"]["betas"]
def run(ctx):
    resolve({"optimizer": {"name": "adamw", "lr": 2e-4}})
''')
fault("code_bug.import", "code_bug", True, "train/__init__.py", '''
def load_plugin():
    from lab_tools import fused_adam_v2
    return fused_adam_v2
def run(ctx):
    load_plugin()
''')
fault("code_bug.zerodiv", "code_bug", True, "train/metrics.py", '''
def throughput(tokens, seconds):
    return tokens / seconds
def run(ctx):
    throughput(1024, 0)
''')
fault("code_bug.recursion", "code_bug", True, "train/config.py", '''
def merge(a, b):
    return merge(b, a)
def run(ctx):
    sys.setrecursionlimit(200)
    merge({}, {})
''')
fault("code_bug.unbound", "code_bug", True, "train/loop.py", '''
def train(steps):
    for step in range(steps):
        loss = 0.1
    return loss
def run(ctx):
    train(0)
''')


# ---- log styles --------------------------------------------------------------------------
def _bar(rng: random.Random, i: int, n: int) -> str:
    pct = int(100 * i / n)
    fill = "█" * (pct // 10) + "▍" * (pct % 10 > 4)
    rate = rng.uniform(0.8, 6.0)
    return f"{pct:3d}%|{fill:<10}| {i}/{n} [{i // 60:02d}:{i % 60:02d}<{(n - i) // 60:02d}:00, {rate:.2f}it/s]"


def style_lines(style: str, rng: random.Random, step: int, total: int, batch: int) -> str:
    loss = 2.6 * (0.995 ** step) + rng.uniform(-0.05, 0.05)
    lr = 2e-4 * min(1, step / 100)
    if style == "hf":
        if rng.random() < 0.3:
            return f"{{'loss': {loss:.4f}, 'grad_norm': {rng.uniform(0.3, 3):.3f}, 'learning_rate': {lr:.2e}, " \
                   f"'epoch': {step / total:.2f}}}"
        return _bar(rng, step, total)
    if style == "lightning":
        return f"Epoch {step * 3 // total}: {_bar(rng, step, total)} loss={loss:.3f} v_num={rng.randint(0, 9)}"
    if style == "plain":
        return (f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | bs {batch} | "
                f"tok/s {rng.randint(8000, 40000)} | mem {rng.uniform(20, 110):.1f}G")
    if style == "datagen":
        return (f"INFO {time.strftime('%m-%d %H:%M:%S')} [loggers.py:{rng.randint(100, 200)}] Engine 000: Avg prompt "
                f"throughput: {rng.uniform(500, 3000):.1f} tokens/s, Avg generation throughput: "
                f"{rng.uniform(50, 900):.1f} tokens/s, Running: {rng.randint(1, 64)} reqs | generated "
                f"{step * 37}/{total * 37} rows")
    if style == "eval":
        return f"[eval] {step}/{total} examples  acc={rng.uniform(0.5, 0.9):.3f}  f1={rng.uniform(0.4, 0.9):.3f}"
    return f"[prep] shard {step % 16}/16 tokenized {step * 1000} rows"


STYLES = ("hf", "lightning", "plain", "datagen", "eval", "prep")

# Lines that mention another failure class's words without being that failure.
DISTRACTORS = [
    "WARNING: expandable_segments not set; memory fragmentation may occur (PYTORCH_CUDA_ALLOC_CONF)",
    "retrying request to http://localhost:8000/v1 after connection reset (attempt 1/5) ... ok",
    "disk usage 82% on /home/lab (41 GB free)",
    "UserWarning: TypedStorage is deprecated. It will be removed in the future.",
    "skipped 3 rows with empty text",
    "NCCL INFO Bootstrap : Using enP7s7:10.0.0.12<0>",
    "WARNING: CUDA initialization: the cuda driver reported version 13.0 (ok)",
    "Token indices sequence length is longer than the specified maximum sequence length (2131 > 2048).",
    "lock acquired after 2.1 s wait: /home/lab/.cache/huggingface/.lock",
    "gradient checkpointing enabled to save memory",
    "FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.",
    "connected to teacher at http://localhost:8000/v1 (model gemma-4-31b-it)",
    "loaded 48210 rows from data/train.jsonl (12 malformed rows dropped)",
]


def _servers(ctx: dict) -> list[object]:
    """Local endpoints for the network faults: a closed port, a silent port and an HTTP 503."""
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    ctx["closed_port"] = closed.getsockname()[1]
    closed.close()  # nothing listens here now

    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen()
    ctx["silent_port"] = silent.getsockname()[1]

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error": "server overloaded"}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    ctx["http503_port"] = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return [silent, srv]


def _load_fault(name: str):
    f = FAULTS[name]
    filename = f"{SITE}/{f.file}"
    src = f.src.lstrip("\n")
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns: dict = {"os": os, "sys": sys, "__name__": f.file.rsplit(".", 1)[0].replace("/", ".")}
    exec(compile(src, filename, "exec"), ns)  # noqa: S102 - our own fault catalogue
    return ns["run"]


def _checkpoints(run_dir: Path) -> list[Path]:
    return sorted((p for p in run_dir.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab.chaos")
    ap.add_argument("--fault", required=True, choices=sorted(FAULTS) + ["none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", choices=STYLES)
    ap.add_argument("--steps", type=int, help="steps to log before the fault (default: random)")
    ap.add_argument("--fail-attempts", type=int, default=1, help="transient faults fire while LAB_ATTEMPT <= this")
    ap.add_argument("--oom-below", type=float, default=0.25, help="oom stops once LAB_BATCH_SCALE <= this")
    ap.add_argument("--run-dir", default=None, help="checkpoint dir (default runs/chaos/<LAB_JOB>)")
    ap.add_argument("--distractors", type=float, default=0.5, help="chance of each distractor slot")
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    style = a.style or rng.choice(STYLES)
    attempt = int(os.environ.get("LAB_ATTEMPT", "1"))
    scale = float(os.environ.get("LAB_BATCH_SCALE", "1"))
    batch = int(os.environ.get("BATCH_SIZE", "64"))  # the healer halves it when the job names it in batch_env
    run_dir = Path(a.run_dir or Path("runs/chaos") / os.environ.get("LAB_JOB", "chaos"))
    run_dir.mkdir(parents=True, exist_ok=True)

    label = FAULTS[a.fault].label if a.fault != "none" else "none"
    fires = {
        "oom": scale > a.oom_below,
        "disk_full": len(_checkpoints(run_dir)) > 2,
        "bad_data": True,
        "code_bug": True,
        "none": False,
    }.get(label, attempt <= a.fail_attempts)

    total = a.steps or rng.randint(40, 400)
    start = 0
    resume = os.environ.get("LAB_RESUME_FROM")
    print(f"python {sys.version.split()[0]} | job {os.environ.get('LAB_JOB', '-')} | attempt {attempt} | "
          f"batch_size {batch}")
    if resume:
        start = int(Path(resume).name.split("-")[1])
        print(f"resuming from {Path(resume).name} (step {start})")
    fault_at = start + max(1, int((total - start) * rng.uniform(0.2, 0.9))) if fires else total
    n_lines = rng.randint(6, 30)
    every = max(1, (fault_at - start) // n_lines)
    distractors = [d for d in DISTRACTORS if rng.random() < a.distractors / 4]
    for step in range(start + 1, fault_at + 1):
        if step % every == 0:
            print(style_lines(style, rng, step, total, batch))
            if distractors and rng.random() < 0.15:
                print(distractors.pop())
        if step % 50 == 0:
            (run_dir / f"checkpoint-{step}").mkdir(exist_ok=True)
            if rng.random() < 0.5:
                print(f"Saving model checkpoint to {run_dir}/checkpoint-{step}")
    for d in distractors:
        print(d)
    sys.stdout.flush()

    if not fires:
        (run_dir / f"checkpoint-{total}").mkdir(exist_ok=True)
        print(f"training finished: {total} steps, final loss {rng.uniform(0.3, 0.9):.4f}")
        return 0

    ctx = {"batch": batch, "step": fault_at, "run_dir": str(run_dir), "alloc": rng.uniform(0.5, 16),
           "free": rng.uniform(0.1, 2.0), "used": rng.uniform(60, 110), "needed": rng.uniform(8, 60)}
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        ctx["tmp"] = tmp
        if label == "network":
            for s in _servers(ctx):
                stack.callback(getattr(s, "server_close", getattr(s, "close", lambda: None)))
        _load_fault(a.fault)(ctx)
    print("fault did not fire", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
