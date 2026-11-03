# spark-lab: rules for Codex and other agents

This repo runs on an NVIDIA DGX Spark (ARM64, 128 GB unified memory). It holds the
job queue, nightly runner and shared tooling for the AI portfolio projects
(scam-radar, autopilot roblox engine, urdu-jailbreak-bench, cricket-shot-coach).
`AGENTS.md` holds the same rules for Codex. Keep the two files identical below the title.

## Ground rules

1. **Training data comes from the Spark.** Every row that trains a model or goes into a
   public dataset is generated or labelled by an open-weight model served on the Spark
   (teacher: Gemma 4 31B on vLLM, `http://localhost:8000/v1`), or is real data collected
   with consent. Never call Claude/Anthropic, OpenAI or Gemini APIs to generate or label
   training data. Evaluating API models is allowed.
2. **`eval/locked/` is human-only.** Never read, edit, list or put its contents in a prompt.
   Only the eval harnesses read it, and only when run by the nightly job.
3. **Uploads are private.** Never run `publish/`. Hugging Face uploads go to private repos;
   the owner makes things public.
4. **Ask before system changes.** Ask before any `sudo`, deleting files outside `runs/`,
   changing systemd units, or touching network/SSH settings.
5. **Long jobs are queue jobs.** Anything longer than a few minutes is a job file in
   `jobs/queue/`, run by the worker service. Training must checkpoint and support resume
   (the worker passes `LAB_RESUME_FROM`).
6. **Every experiment is reproducible.** A config file per experiment, committed; results
   appear in the nightly report; no hand-edited numbers in READMEs.
7. **No personal data leaves the Spark.** Anonymise before any upload or commit.

## Daily loop

Threads write code and open pull requests. The owner merges. Each night
`jobs/nightly.sh` pulls `main`, enqueues `jobs/nightly.d/*.json`, runs the queue until
empty or the deadline, writes `reports/<date>.md` and commits it. Read the latest report
before changing anything.

## Useful commands

```bash
python -m lab status                 # queue, running job, disk, GPU
python -m lab enqueue jobs/nightly.d/example.json
python -m lab worker --once          # run one job in the foreground
python -m lab report --date today    # write the nightly report
python -m lab prune runs/ --keep 3   # keep the newest 3 checkpoints per run
pytest -q
```
