"""Real GPU chaos jobs for the Spark: faults the CPU cloud session could only emulate.

    python -m labml.chaos_gpu --fault oom

Run through the queue with `"heal": false` (see jobs/chaos/), then
`python -m labml.logs collect` turns the failed jobs into labelled rows and
`python -m labml.logs eval-real` scores the shipped classifier on them.

| --fault          | label      | how                                                                 |
| ---------------- | ---------- | ------------------------------------------------------------------- |
| oom              | oom        | cap this process at 2% of GPU memory, then allocate past it        |
| oom_backward     | oom        | same cap, a real forward/backward with a batch that does not fit   |
| device_assert    | bad_data   | cross-entropy with a label >= n_classes on the GPU                  |
| illegal_access   | cuda_error | Triton kernel storing far outside its buffer (may need adjusting)   |
| dataloader_kill  | oom        | DataLoader worker killed with SIGKILL, as the OOM killer would      |
| vllm_kv          | oom        | vLLM with gpu_memory_utilization too small for the model            |

Safety: the OOM faults cap the process with `set_per_process_memory_fraction`, so they
never press on the Spark's unified memory (which the teacher and the OS share). Run them
when the teacher is idle anyway. `dataloader_kill` works on CPU too (tested in the cloud).
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time


def _preamble(seed: int, steps: int = 40) -> None:
    from lab.chaos import style_lines

    rng = random.Random(seed)
    style = rng.choice(("hf", "plain", "lightning"))
    for step in range(1, steps + 1):
        if step % 4 == 0:
            print(style_lines(style, rng, step, 200, 32))
    sys.stdout.flush()


def oom(backward: bool) -> None:
    import torch

    torch.cuda.set_per_process_memory_fraction(0.02)
    if not backward:
        blocks = []
        while True:
            blocks.append(torch.empty(256 << 20, dtype=torch.uint8, device="cuda"))
    model = torch.nn.Sequential(torch.nn.Linear(4096, 16384), torch.nn.GELU(), torch.nn.Linear(16384, 4096)).cuda()
    x = torch.randn(65536, 4096, device="cuda")
    model(x).pow(2).mean().backward()


def device_assert() -> None:
    import torch

    logits = torch.randn(64, 10, device="cuda")
    target = torch.randint(0, 10, (64,), device="cuda")
    target[7] = 12  # a label outside the 10 classes, as a bad row would have
    loss = torch.nn.functional.cross_entropy(logits, target)
    torch.cuda.synchronize()
    print(float(loss))


def illegal_access() -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def scatter(ptr, n, BLOCK: tl.constexpr):  # noqa: N803
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(ptr + offs * (1 << 28), 1.0, mask=offs < n)

    buf = torch.zeros(16, device="cuda")
    scatter[(4,)](buf, 1 << 12, BLOCK=1024)
    torch.cuda.synchronize()


def dataloader_kill() -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Slow(Dataset):
        def __len__(self):
            return 10_000

        def __getitem__(self, i):
            if i == 300:
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.001)
            return torch.zeros(4)

    for _ in DataLoader(Slow(), batch_size=32, num_workers=2):
        pass


def vllm_kv() -> None:
    from vllm import LLM

    LLM(model=os.environ.get("TEACHER_MODEL", "google/gemma-4-31b-it"), gpu_memory_utilization=0.08,
        max_model_len=8192, enforce_eager=True)


FAULTS = {"oom": ("oom", lambda: oom(False)), "oom_backward": ("oom", lambda: oom(True)),
          "device_assert": ("bad_data", device_assert), "illegal_access": ("cuda_error", illegal_access),
          "dataloader_kill": ("oom", dataloader_kill), "vllm_kv": ("oom", vllm_kv)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.chaos_gpu")
    ap.add_argument("--fault", required=True, choices=sorted(FAULTS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(f"python {sys.version.split()[0]} | job {os.environ.get('LAB_JOB', '-')} | gpu chaos")
    _preamble(a.seed)
    FAULTS[a.fault][1]()
    print("fault did not fire", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
