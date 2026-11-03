# spark-lab

The always-on job runner behind my AI projects. An NVIDIA DGX Spark generates data,
fine-tunes models and runs evals every night; coding agents write the code by day; a
report lands in `reports/` every morning.

```
pull requests ──► main ──► 23:00 nightly.sh ──► queue ──► worker ──► reports/<date>.md ──► phone alert
                                   ▲                 │
                         jobs/nightly.d/*.json       └─ checkpoints, retries, resume after power cuts
```

## What it does

- **File-based job queue** (`lab/queue.py`). Jobs are JSON files that move
  `queue/ → running/ → done/ | failed/` with atomic renames, so a crash or power cut never
  loses one. On restart, interrupted jobs go back to the front of the queue.
- **Resume, not restart.** Jobs with `resume_dir` get the newest `checkpoint-N/` passed
  as `LAB_RESUME_FROM` on every attempt, so a power cut costs one checkpoint interval.
- **Retries, timeouts, a single-worker lock, and a disk guard** that pauses the worker
  and alerts before the disk fills.
- **Phone alerts through ntfy** for job done, job failed (with the last log lines),
  disk nearly full, and the nightly summary.
- **Nightly report** (`lab/report.py`): jobs from the last 24 hours, results quoted from
  each job's own markdown report, failures, disk and GPU. Refuses to quote anything
  inside `eval/locked/`.
- **systemd units** for the worker, the 23:00 nightly timer (catches up after a missed
  night) and the vLLM teacher model.

## Quickstart on the Spark

```bash
mkdir -p ~/lab && cd ~/lab
git clone git@github.com:JehanzebAli11/spark-lab.git && cd spark-lab
bash scripts/setup_spark.sh            # checks only; lists the sudo steps for you to run
bash scripts/setup_spark.sh --apply    # installs the user services
python3 -m lab notify "hello"          # test the phone alert
python3 -m lab enqueue jobs/nightly.d/example-smoke.json && python3 -m lab worker --once
```

Project repos sit next to this one (`~/lab/scam-radar`, …) and jobs reach them with
`"cwd": "../scam-radar"`. See `jobs/examples/` for a training job that resumes from checkpoints and an eval job whose markdown lands in the nightly report.

## Job format

```json
{
  "name": "scam-radar-sft-v1",
  "cmd": ["python3", "-m", "scamradar", "train", "--config", "scamradar/train/configs/v1.yaml"],
  "cwd": "../scam-radar",
  "priority": 20,
  "timeout_min": 240,
  "retries": 2,
  "resume_dir": "runs/v1",
  "report": "runs/v1/eval.md"
}
```

Lower priority runs first. `report` is a markdown file the job writes; the nightly report
quotes it.

## Commands

| Command | Does |
| --- | --- |
| `python3 -m lab enqueue FILE...` | Validate and queue job files |
| `python3 -m lab worker [--once \| --drain] [--until HH:MM]` | Run jobs |
| `python3 -m lab status` | Queue, running job, disk, GPU |
| `python3 -m lab report [--date YYYY-MM-DD]` | Write `reports/<date>.md` |
| `python3 -m lab prune runs/ --keep 3 [--dry-run]` | Keep the newest checkpoints per run |
| `python3 -m lab notify "text"` | Test phone alerts |

## Ground rules

Training data comes only from open-weight models on the Spark or real consented data,
never from Claude, OpenAI or Gemini. `eval/locked/` is human-only. Uploads are private
until I publish them by hand with `publish/publish.sh`. Full rules for coding agents are
in [`CLAUDE.md`](CLAUDE.md) and [`AGENTS.md`](AGENTS.md).

## Development

```bash
pip install -e '.[dev]' && ruff check . && pytest -q
```

Standard library only at runtime, Python 3.10+. MIT licensed.
