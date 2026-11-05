# HANDOFF: self-healing queue, chaos log classifier, training-run doctor

Branch `self-healing`, draft PR https://github.com/JehanzebAli11/spark-lab/pull/2.
Written by a cloud session (4 CPU cores, no GPU, no Hugging Face, no API keys) for the
owner to continue with Claude Code on the DGX Spark (`~/lab/spark-lab`, ARM64 + CUDA,
128 GB unified memory). Every number below comes from a committed result file; none is
typed by hand. If this file looks unfinished, credits ran out mid-way: the status table
and `git log` are the source of truth.

## 1. Status

### Done and verified here (with result files)

| What | Evidence |
| --- | --- |
| Self-healing queue: classify each failed attempt, apply fix, incidents, report section | `lab/heal.py`, `lab/triage.py`, `lab/queue.py`; `tests/test_heal.py` runs real chaos jobs through the real worker (39 tests pass, `pytest -q`) |
| A night of 8 chaos jobs through the real worker: 4 healed, 4 incidents | `results/demo/queue_report.md`, `results/demo/queue.gif`, `results/demo/example_incident.md` |
| Chaos fault catalogue: 75 variants, 34 of them real faults | `lab/chaos.py`, `datasets/chaos-logs/README.md` |
| Labelled log dataset, 4,500 rows, built through `Queue.run` | `datasets/chaos-logs/logs.jsonl.gz` |
| Log classifier: TF-IDF + logistic regression exported to stdlib JSON, threshold escalates | `lab/models/log_classifier.json`, `results/log_classifier/report.md` |
| Its iteration history (the held-out split was looked at; honest caveat) | `results/log_classifier/ITERATIONS.md` |
| Evidence guard: logs with no error line, signal exit or rule hit -> `unknown` | `lab/triage.py:has_evidence`, `tests/test_heal.py::test_threshold_escalates_low_confidence` |
| Doctor dataset: 4,410 tiny numpy runs, 9 causes, references, curve images | `datasets/doctor-runs/`, `labml/doctor/sim.py`, `dataset.py` |
| Doctors: image CNN, curve GBM, features GBM, ensemble; kNN novelty; leave-one-cause-out | `results/doctor/report.md`, `metrics.json`, confusion PNGs, `curve_grid.png` |
| Auto-fix loop over 780 harmed runs, vs plain rerun and true-cause oracle | `results/doctor/autofix.md`, `autofix.json`, `autofix_runs.jsonl`, `rescue_by_cause.png`, `results/demo/doctor.gif` |
| Triage HTTP server + `LAB_TRIAGE_URL` client with local fallback, `lab classify` CLI | `lab/serve.py`, `tests/test_heal.py::test_triage_server_and_remote_client` |
| README results block generated from the JSON results | `python -m labml.readme --check` |
| One real PyTorch fault on CPU (DataLoader worker SIGKILL, `labml/chaos_gpu.py --fault dataloader_kill`) through queue -> `collect` -> `eval-real`: 2/2 classified `oom` (0.90, 0.94) | run by hand in the session, not committed as a result (2 rows is an anecdote) |

Headline numbers (from the result files):

- Log classifier, unseen fault variants: 0.683 accuracy without threshold; with the 0.55
  threshold it acts on 50.5% at 0.717 precision and escalates the rest. All 60 wrong
  actions come from one variant (`disk_full.sqlite` read as `flaky`, cost: 2 plain
  retries, then an incident). In-distribution test: 1.000 (easy: same phrasings).
- Doctor test accuracy: features 0.984, ensemble 0.982, curve numbers 0.910, image-only
  CNN 0.893; on the unseen `rings` task family 0.969 / 0.962 / 0.886 / 0.824.
- Unseen causes (leave-one-out): with the novelty check, 5 of 8 causes are mostly
  escalated; `label_noise` and `train_test_leak` get called healthy (~72-73%), and
  `lr_too_low` gets called `label_noise`.
- Auto-fix: features doctor rescues 458/780 harmed runs (58.7%); rerunning unchanged
  rescues 10/780; the true-cause fix 536/780. The doctor touched 8/130 healthy runs and
  made 3 worse by > 2 points. `label_noise` is rescued 0/130 even by the oracle
  (small-loss filtering gains +3.3 points of a ~21-point gap). For `no_warmup` the doctor
  beats the oracle (20 vs 8 of 29): adding warmup rarely helps, lowering the LR does.

### Done but tested only against fakes or emulations

| What | Why not verified |
| --- | --- |
| GPU fault classes (`cuda_error`, GPU OOM, NCCL, cuDNN, vLLM, Xid) | No GPU: these chaos variants raise exceptions with copied library messages. Real GPU logs may be worded differently. |
| `labml/chaos_gpu.py` faults `oom`, `oom_backward`, `device_assert`, `illegal_access`, `vllm_kv` and `jobs/chaos/gpu-*.json` | Need CUDA. `illegal_access` (Triton out-of-bounds store) may not trigger as written. |
| VLM doctor (`labml/doctor/vlm.py`) | Tested against a fake OpenAI-compatible server only (`tests/test_vlm.py`). Never seen a real model. |
| `lab-triage.service`, `setup_spark.sh` enabling it | systemd not available here; unit only syntax-checked by reading. |
| Healer on real training jobs (scam-radar etc.) | Needs the project repos and `batch_env` set in their job files. |

### Not started

- Doctor on real training runs (needs a metrics adapter from each project's logs to `series`).
- Relabelling noisy data with the Gemma teacher as the `label_noise` fix.
- Feeding Spark-collected real failures back into the classifier's training data.

## 2. Spark setup

```bash
cd ~/lab/spark-lab
git fetch origin
git checkout self-healing        # to try before merging; after you merge PR #2: git checkout main && git pull
python3 -m pytest -q             # needs only the standard library + pytest; expect "39 passed"
```

ML environment (only for `labml/`; the worker keeps using system `python3`):

```bash
python3 -m venv ~/lab/.venv-ml
~/lab/.venv-ml/bin/pip install -U pip
# PyTorch with CUDA on ARM64: use NVIDIA's wheel for the Spark's CUDA version. Check the DGX Spark playbook;
# one option is: ~/lab/.venv-ml/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
~/lab/.venv-ml/bin/pip install -e '.[ml,dev]'
~/lab/.venv-ml/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Expected: `True NVIDIA GB10` (or similar). If `is_available()` is False, stop and fix
PyTorch before any GPU step.

Services (user units, no sudo; `--apply` copies all units and enables worker, timer, triage):

```bash
bash scripts/setup_spark.sh --apply
systemctl --user restart lab-worker.service     # picks up the healer
curl -s localhost:8040/health                   # {"ok": true, ...}
python3 -m lab classify logs/<some failed job>.log
```

The worker unit runs `/usr/bin/env python3 -m lab worker`: the healer and classifier are
standard library, so nothing else is needed for healing. Optional in `~/.config/spark-lab.env`:
`LAB_TRIAGE_URL=http://127.0.0.1:8040` (the worker then asks the server, falls back locally).

## 3. Remaining work, in order

### Step 1. Verify the healer on the Spark (10 min)

```bash
python3 -m lab enqueue jobs/chaos/heal-*.json && python3 -m lab worker --drain
python3 -m lab report && sed -n '/Self-healing/,/Failures/p' reports/$(date +%F).md
```

Expected: `heal-oom`, `heal-network`, `heal-flaky` done (network waits 30 s + 60 s for
real); `heal-code-bug` failed with an incident in `incidents/`. Same table as
`results/demo/queue_report.md`.

### Step 2. Real GPU chaos jobs (1-2 h, mostly waiting)

Run when the teacher is idle (`systemctl --user stop vllm-teacher.service` for `vllm_kv`,
then start it again).

```bash
python3 -m lab enqueue jobs/chaos/gpu-*.json && python3 -m lab worker --drain
ls jobs/failed/ | grep chaos-gpu        # expect 16 failed jobs (they are meant to fail)
~/lab/.venv-ml/bin/python -m labml.logs collect
~/lab/.venv-ml/bin/python -m labml.logs eval-real   # writes results/log_classifier/real_eval.md
```

Job file example (`jobs/chaos/gpu-oom-0.json`):

```json
{"name": "chaos-gpu-oom-0", "cmd": ["../.venv-ml/bin/python", "-m", "labml.chaos_gpu", "--fault", "oom", "--seed", "0"],
 "priority": 85, "timeout_min": 5, "retries": 0, "heal": false, "notify": false,
 "env": {"LAB_CHAOS_LABEL": "oom", "LAB_CHAOS_FAULT": "gpu.oom"}}
```

Check each log by eye: does `oom` end in `torch.OutOfMemoryError: CUDA out of memory`?
Does `device_assert` show `device-side assert triggered`? If `illegal_access` printed
"fault did not fire", adjust the Triton offset or drop it. Expected: the shipped model gets
most real GPU OOMs right (emulated message was copied from PyTorch); real CUDA errors are
the open question. Commit `real_eval.md` and `spark_real.jsonl.gz` (check it holds no
personal paths or data first: `zcat datasets/chaos-logs/spark_real.jsonl.gz | grep -i home`).

### Step 3. Retrain the classifier with real GPU logs (20 min)

Add a `spark_real` pathway to `labml/logs.py:train`: keep half the real rows (by fault) as
a new test split, add the rest to train, rerun `python -m labml.logs train`, then
`python -m labml.readme`. Report the real test split separately. To keep adding real
failures from normal nights, extend `collect` to also read a hand-kept `labels.jsonl` of
`{"log": "logs/<file>.log", "label": "<class>"}` written when the owner reviews incidents.

### Step 4. VLM doctor via the Spark's Gemma (30-60 min)

The teacher must accept images: vLLM needs the multimodal model and
`--limit-mm-per-prompt '{"image": 1}'` (check `systemd/vllm-teacher.service`).

```bash
curl -s localhost:8000/v1/models
~/lab/.venv-ml/bin/python -m labml.doctor.vlm --url http://localhost:8000/v1 --model google/gemma-4-31b-it --limit 270
```

Job file (`jobs/nightly.d/` or one-off):

```json
{"name": "doctor-vlm-eval", "cmd": ["../.venv-ml/bin/python", "-m", "labml.doctor.vlm", "--limit", "270"],
 "priority": 60, "timeout_min": 120, "retries": 1, "report": "results/doctor/vlm.md"}
```

Expected: `results/doctor/vlm.md` with accuracy per split and recall per cause, ~1-3 s per
image. Compare with the image CNN (0.893 test). If Gemma does well on some causes, a
cheap next step is a VLM + features ensemble. Its answers are evaluation, not training
data, unless you decide to use the teacher's labels.

### Step 5. Serve the classifier from the Spark (10 min)

Done by `setup_spark.sh --apply`. To use it from another machine on the tailnet, change
`--host` in `systemd/lab-triage.service` to the Spark's Tailscale IP (never 0.0.0.0; no
auth), then on the other machine `LAB_TRIAGE_URL=http://spark-1:8040 python3 -m lab classify x.log`.

### Step 6. Point real jobs at the healer (30 min per project)

In each project's job JSON add `"batch_env": "BATCH_SIZE"` with `"env": {"BATCH_SIZE": "..."}`
and make the trainer read that variable (and `LAB_BATCH_SCALE` if it computes its own
batch), plus `resume_dir` so halving resumes from a checkpoint. Watch the first nights'
Self-healing sections.

### Step 7. Doctor on real runs (2-4 h)

Write an adapter from HF Trainer `trainer_state.json` (`log_history`) to the doctor's
`series` (interpolate to 50 points; grad_norm and learning_rate are logged by HF), run the
features doctor on finished runs, add a "Doctor" line per training job to the nightly
report. Expect a domain gap: the doctor was trained on tiny MLPs.

## 4. Risks and what to verify first on real hardware

1. **GPU error wording.** Emulated messages may not match the PyTorch/driver/vLLM versions
   on the Spark. Step 2 measures this; do it before trusting `cuda_error` decisions.
2. **Unified memory and OOM.** On the Spark, GPU and CPU share 128 GB. A real OOM may
   surface as the kernel OOM killer (SIGKILL, exit -9) rather than a CUDA OOM, and may hit
   the teacher. The healer treats SIGKILL as `oom` only if the model says so; check
   `dmesg` after the first real OOM. `chaos_gpu.py` caps itself at 2% of GPU memory.
3. **Halving the batch only works if the trainer reads `batch_env`.** Otherwise the retry
   repeats the OOM 3 times then writes an incident (wasted time, not harm).
4. **Pruning.** `disk_full` deletes all but the newest 2 checkpoints in any `runs/` tree the
   job uses (CLAUDE.md allows deletes under `runs/`). If a run needs an older checkpoint
   (best-val), keep it outside `checkpoint-N/` naming.
5. **Back-off blocks the single worker** (up to 30+60+120+240 s per network failure).
   Acceptable at night; revisit if many jobs hit network errors.
6. **Wrong-but-confident classifier.** On unseen phrasings precision when acting was 0.717.
   The budgets cap the cost (worst cases: 2 needless retries for `flaky`, 3 halvings for
   `oom`). Read incident files for the first week and add mislabelled cases to the data.
7. **`incidents/` is gitignored** (log tails may hold personal data, CLAUDE.md rule 7);
   agents read them locally.

## 5. Prompt for Claude Code on the Spark

```text
You are on the DGX Spark in ~/lab/spark-lab. Read CLAUDE.md, README.md and HANDOFF.md
first; HANDOFF.md is the plan and says what is verified, what was only tested against
fakes, and what is not started. Check out the `self-healing` branch (or main if PR #2 is
merged) and run `python3 -m pytest -q` (expect 39 passed).

Then work through HANDOFF.md section 3 in order, starting at Step 1. For each step: run
the commands, compare with the expected output, commit result files, and update the
status table in HANDOFF.md (move items from "tested only against fakes" to "verified"
with the result file as evidence). Long work goes through the queue as job files
(CLAUDE.md rule 5). Ask before any sudo, systemd unit change outside `systemctl --user`,
or deleting files outside runs/. Never use Claude/OpenAI/Gemini output as training data;
the Gemma teacher on localhost:8000 is the allowed teacher. Never invent numbers: every
number in a README or HANDOFF must come from a file the code wrote
(`python -m labml.readme` regenerates the README block). Before Step 2, stop the teacher
if it is serving and start it again afterwards. Read section 4 (risks) before Step 2 and
check dmesg after the first real OOM. Open pull requests; the owner merges.
```

## Stretch list (this session, in order)

- [x] Self-healing queue, chaos dataset, classifier, doctor, auto-fix, demos, README
- [x] Triage server + remote client + CLI, systemd unit
- [x] VLM doctor client (fake-tested), GPU chaos script + job files, collect / eval-real
- [x] CI: `python -m labml.readme --check` (the `ml` extra is not in CI; training is too slow for it)
- [ ] Doctor: a stronger label-noise fix (co-teaching) and a two-cause dataset
- [ ] Classifier: per-class thresholds; calibrate with temperature scaling
- [ ] Nightly report: a one-line doctor verdict per training job (needs Step 7 adapter)
