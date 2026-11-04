# Handover

State of spark-lab as of 2026-11-04, for whoever picks this up next (a person or a coding
agent). Read `CLAUDE.md` first: its rules override anything here.

## What this repo is

The always-on job runner for the AI portfolio projects (scam-radar, the Roblox autopilot engine,
urdu-jailbreak-bench, cricket-shot-coach). It runs on an NVIDIA DGX Spark (ARM64, GB10, 128 GB
unified memory).

1. Coding agents open pull requests during the day; the owner merges.
2. `jobs/nightly.sh` runs at 23:00. It pulls `main`, queues `jobs/nightly.d/*.json` and runs the
   queue until 08:30.
3. It writes `reports/<date>.md`, commits it, and sends a phone alert through ntfy.

The Gemma 4 31B teacher on vLLM (`localhost:8000`) generates and labels all training data.

| Folder | State |
| --- | --- |
| `lab/` | Queue, worker, reports, checkpoints, ntfy alerts, teacher smoke test. Working and tested. |
| `jobs/` | `nightly.sh`, the smoke job, and examples for scam-radar train and eval. |
| `systemd/` | User units for the worker, nightly timer and vLLM teacher, plus the env example. |
| `scripts/` | `bootstrap.sh` (first-time setup) and `setup_spark.sh` (read-only health check). |
| `publish/` | Owner-only upload to private Hugging Face repos. Agents never run it. |
| `factory/`, `train/`, `eval/harness/` | Empty placeholders. |
| `eval/locked/` | Human-only. Agents must never read, list or prompt with it. |

## Work done in this session

**Branch `bootstrap`, PR [JehanzebAli11/spark-lab#1](https://github.com/JehanzebAli11/spark-lab/pull/1)**
(open, not merged):

- **`scripts/bootstrap.sh`**: a guided, idempotent first-time setup. Details in
  [`docs/bootstrap.md`](bootstrap.md). It never runs sudo: it prints each root command and waits
  while the owner runs it. It writes `~/.config/spark-lab.env`, asking only for the ntfy topic and
  HF token, then:
  - pulls the images and downloads Gemma 4 31B
  - installs the user services and waits for `/v1/models`
  - runs a 3-prompt Roman Urdu smoke test and queues the smoke job
  - prints a checklist
- **`python3 -m lab teacher-smoke`** (`lab/teacher.py`): the smoke test, using only the standard
  library.
- **`systemd/vllm-teacher.service`**: now starts with `--entrypoint vllm … serve`, so it works
  with both `vllm/vllm-openai` and NVIDIA's NGC images.
- **Env example fixes:**
  - `TEACHER_MODEL=google/gemma-4-31B-it`, with a capital B: the Hugging Face ID is case-sensitive.
  - `VLLM_IMAGE` set to the verified image (next section).
  - `VLLM_GPU_UTIL` raised from 0.55 to 0.70.
- **Tests:** `tests/test_bootstrap.py` runs the script end to end twice, against stub
  docker/systemctl/sudo and a fake OpenAI server (`tests/fake_openai.py`). `tests/test_teacher.py`
  covers the smoke test.
- **CI** now runs shellcheck; `shellcheck-py` is in the dev extras.
- **Locally:** ruff and shellcheck are clean, and 51 tests pass.

**Not yet tested on a real Spark.** The first real run is the real test. Watch
`journalctl --user -u vllm-teacher | grep -i 'KV cache'`.

## Facts verified (2026-11-03), sources in `docs/bootstrap.md`

- **Teacher image:** `vllm/vllm-openai:gemma4-cu130`, pinned by digest `sha256:0d1525…`. NVIDIA's
  DGX Spark playbook names it for Gemma 4; it is CUDA 13.0 with an ARM64 build.
- **Why not NGC:** NGC `nvcr.io/nvidia/vllm` tags from 26.03 on need driver ≥ 595.58, and DGX OS
  ships 580.x.
- **Unsloth:** NVIDIA's playbook uses `nvcr.io/nvidia/pytorch:25.11-py3` with
  `pip install --no-deps unsloth unsloth_zoo bitsandbytes`.
- **Gemma 4 31B (BF16):** 62.5 GB of weights, Apache-2.0, not gated. Expect about 4 tok/s per
  sequence, so batch the data generation. NVFP4 (`nvidia/Gemma-4-31B-IT-NVFP4`, 32.6 GB, NVIDIA
  licence) is about 2× faster.

## Project direction (brainstorm, not decided)

The full write-up is in the Claude doc
["Building our own System One models"](https://claude.ai/artifact/2oYdKqZiLcvq3ocPCLTJKP).
It covers how TypeSafe's Jev works and how to build an open equivalent ("Spark-1") on the Spark.

Recommended order:

1. **Spark-1:** a small Gemma 4 E2B fine-tuned to return typed decisions with calibrated
   probabilities (Choice, Noul, Score, Box) over text and images.
   - Labels: soft labels from the teacher's logprobs, plus procedural ground truth.
   - Calibration: temperature scaling.
2. **"Knows when it's wrong" calibration benchmark:** Spark-1 against Jev, and against frontier
   LLMs through OpenRouter (evaluation only).
3. **Demos:**
   - a System One / System Two reflex game agent (in your own Roblox Studio game, not live Roblox)
   - a Roman Urdu guardrail
   - a screenshot scam judge

**Other VLM ideas, where Hugging Face searches found nothing:**
- **Vision Scope:** sparse autoencoders on Gemma 4's vision tokens.
- **Blind Spot Lab:** procedural tests that frontier VLMs fail.
- **Pixel Pilot:** a game agent that plays from screen frames.
- **Dream Arcade:** a tiny playable world model.
- **Sketch2Diagram:** whiteboard photo to an editable diagram.

**Pakistan-specific gaps:**
- an Urdu scam/fraud dataset
- an Urdu jailbreak benchmark
- Urdu scene text
- a bowling-action checker

## Rules that shape all of the above

- Training data comes only from the Spark teacher or from real data collected with consent.
- Claude, OpenAI, Gemini and OpenRouter models may be evaluated, never used to make training data.
- Claude credits are for coding agents.
- Uploads stay private until the owner publishes them by hand.

## Open questions for the owner

1. May public human-labelled datasets (NLI, moderation, VQA) count as ground truth under rule 1?
2. Should we pay for Jev API calls to run the comparison? It costs $0.042 per million input tokens.
3. Which first project: Spark-1, or Blind Spot Lab (the faster win)?

## Next steps

1. Review and merge PR #1, then run `bash scripts/bootstrap.sh` on the Spark over SSH.
2. Check the first nightly report and the teacher's real throughput.
3. Pick the first project, then add its generators to `factory/`, its training to `train/` and
   its eval to `eval/harness/`, each as queue jobs that checkpoint.
