# Dataset card: chaos-logs

Log tails of failed jobs, each labelled with the fault that was injected into it. Built to
train and test the self-healing queue's failure classifier (`lab/triage.py`).

## Summary

| | |
| --- | --- |
| File | `logs.jsonl.gz` (4,500 rows, ~1 MB) |
| Classes | `oom`, `disk_full`, `cuda_error`, `network`, `bad_data`, `flaky`, `code_bug` |
| Fault variants | 75 (`lab/chaos.py`, `FAULTS`), 9–14 per class |
| Seeds per variant | 60 |
| Real faults | 2,040 rows (34 variants); the rest emulated, see below |
| Label source | code: the fault the chaos job was told to inject. No model labelled anything. |
| Licence | MIT, same as the repo |

## How it was made

```bash
python -m labml.logs build --seeds 60
```

Each row is one real job: `lab.chaos` is enqueued and run by the real `Queue.run` (healing
off, no retries) in a throwaway lab root, so the log format, the `=== exit N` line and the
tail extraction are exactly what the healer sees in production. Each chaos job prints a
training-style log (6 styles: HF Trainer, Lightning, plain loop, vLLM data generation, eval
harness, preprocessing) with random distractor lines that mention other classes' words
(a memory warning before a network error, a recovered connection reset before an OOM),
then hits its fault.

**Real faults** (actually happen in the process): `MemoryError` under an address-space
limit, `SIGKILL` (as the kernel OOM killer does), `ENOSPC` from `/dev/full`, `EDQUOT`,
refused connections, DNS failures, socket timeouts, HTTP 503 from a local server, broken
pipes, held `flock`s, `EADDRINUSE`, locked SQLite, `ETXTBSY`, `EAGAIN`, truncated JSON,
bad UTF-8, corrupt gzip and pickle, missing columns, real Python bugs.

**Emulated faults** (cannot happen on a CPU-only box): CUDA, cuDNN, cuBLAS, NCCL, Xid,
GPU OOM, vLLM, JAX/XLA, TensorFlow, Triton, and a few library errors (`requests`, `httpx`,
`pyarrow`, `PIL`, `datasets`, `docker`). These raise an exception whose module, type and
message are copied from the real library, from code compiled under a realistic
site-packages filename, so the traceback has the real shape. They have not been checked
against real GPU failures yet; that is the first Spark task in `HANDOFF.md`.

Tracebacks never mention the label: fault code runs as `run()`/`forward()`/`save()`-style
functions in files like `torch/nn/functional.py`. The queue's attempt header (which holds
the command line, and so the fault name) is excluded from the text by `lab.triage.tail`.

## Fields

| Field | Meaning |
| --- | --- |
| `id` | `<fault>:<seed>` |
| `fault` | variant name, e.g. `oom.torch_cuda` |
| `label` | class (the part before the dot) |
| `real` | true if the fault really happened, false if emulated |
| `style` | log style of the job body |
| `seed` | chaos seed |
| `split` | see below |
| `returncode` | the job's exit code (1, or -9 for SIGKILL) |
| `text` | last 60 lines of the attempt, as `lab.triage.tail` returns them |

## Splits

| Split | Rows | What |
| --- | --- | --- |
| train | 2,562 | 61 in-distribution variants, seeds 0–41 |
| val | 549 | same variants, seeds 42–50 |
| test | 549 | same variants, seeds 51–59 |
| heldout_val | 420 | 7 variants (one per class) never in train; used to tune C and the threshold |
| heldout_test | 420 | 7 more unseen variants; reported only |

The in-distribution test is nearly trivial (same phrasings, new seeds). The held-out
splits are the real measure: they ask whether the classifier recognises a failure it has
never seen worded that way.

## Privacy

Machine paths are rewritten to `/home/lab/...` and temp dirs to `/tmp/job`. The data is
synthetic: no personal data, no model outputs.

## Known limits

- One fault per log. Real logs sometimes show a cascade (an OOM that surfaces as a NCCL
  timeout); those would be labelled by their root cause and are not covered.
- `flaky` is defined by the fix (a plain retry works), which a log cannot always show.
- The emulated GPU messages are the library's current wording; versions on the Spark may
  differ.
