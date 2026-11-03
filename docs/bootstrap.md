# First-time Spark setup: `scripts/bootstrap.sh`

One guided run, over SSH, as your normal user, from `~/lab/spark-lab`:

```bash
bash scripts/bootstrap.sh
```

It is idempotent: rerun it any time, and finished steps are skipped. It never runs `sudo`.
When a step needs root it prints the exact command, waits while you run it in a second SSH
session, then re-checks. You can also type `skip`; a skipped step ends up in the final checklist.

| # | Step | What it does | Skipped on rerun when |
|---|------|--------------|-----------------------|
| 1 | Machine | aarch64, GPU and driver (≥ 580 for CUDA 13.0), checkout at `~/lab/spark-lab`, refuses to run as root | always runs (read-only) |
| 2 | sudo steps | docker group, linger, user manager in the docker group, mask sleep, tmux, `agent` user, SSH keys only | each check passes |
| 3 | Config | writes `~/.config/spark-lab.env` (mode 600) from `systemd/spark-lab.env.example`; asks **only** for the ntfy topic (offers a random one) and the HF read token (hidden input, checked against `/api/whoami-v2`) | values already set |
| 4 | Images | `docker pull` of the teacher image and the Unsloth training base image | image present |
| 5 | Weights | downloads `google/gemma-4-31B-it` into `~/.cache/huggingface`, using the teacher image's `huggingface_hub`, as your uid | files already cached (HF checks and skips) |
| 6 | Services | copies `systemd/*` into `~/.config/systemd/user`, `enable --now` the timer, worker and teacher; restarts a running service only if its unit or the env file changed | units unchanged |
| 7 | Teacher | polls `http://localhost:8000/v1/models` until the model is listed (default 45 min; stops early if the service fails, printing its log) | — |
| 8 | Smoke test | `python3 -m lab teacher-smoke`: 3 Roman Urdu prompts; each answer must be in Latin script with Urdu function words. Answers are printed, never saved | — |
| 9 | Smoke job | enqueues `jobs/nightly.d/example-smoke.json` and waits up to 2 min for the worker to run it | a smoke job is already in `queue/`, `running/` or `done/` |
| 10 | Checklist | ticks what is done, lists what is open, plus the by-hand items (phone alert, UPS, timer, …) | — |

Exit codes: `0` done (maybe with open items), `1` a step failed, `3` you were just added to the
docker group and need to SSH in again.

Knobs (environment variables, mostly for tests): `TEACHER_WAIT_MIN`, `TEACHER_URL`,
`MODEL_MIN_FREE_GB`, `BOOTSTRAP_POLL_SEC`, `BOOTSTRAP_JOB_WAIT_SEC`, `BOOTSTRAP_NOTIFY=0`.

`scripts/setup_spark.sh` stays as a quick read-only health check.

## Research (checked 2026-11-03)

### Teacher image: `vllm/vllm-openai:gemma4-cu130`

- NVIDIA's **DGX Spark vLLM playbook** ([build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm),
  source: [NVIDIA/dgx-spark-playbooks `nvidia/vllm`](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm),
  commit `3088315`, "Last Updated 06/12/2026") lists `google/gemma-4-31B-it` (and the NVFP4
  variant) as supported on Spark and says, verbatim: *"For Gemma 4 model family, use vLLM custom
  container: `docker pull vllm/vllm-openai:gemma4-cu130`"*. Its generic NGC example is
  `nvcr.io/nvidia/vllm:26.05.post1-py3`, and its cluster steps pin `nvcr.io/nvidia/vllm:26.05-py3`.
- [Docker Hub `vllm/vllm-openai:gemma4-cu130`](https://hub.docker.com/r/vllm/vllm-openai/tags?name=gemma4-cu130):
  last pushed 2026-04-10, index digest
  `sha256:0d152595cd940ea1e0890fa190a37b7a86e3f7f0f5048ac3e2e4c14529fea833`, with an `arm64` image
  (`sha256:2970dd3c…`) next to `amd64`. `cu130` = CUDA 13.0, which runs on the 580-series driver DGX
  OS ships. We pin `tag@digest`, so a re-pushed tag can't change the teacher under us.
- Why not the newest NGC tag: `nvcr.io/nvidia/vllm:26.10-py3` exists
  ([NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm),
  [release notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)), but
  NGC tags from 26.03 on are built for driver ≥ 595.58 and fail the forward-compat check on the
  Spark's 580 driver; NVIDIA staff confirmed in August 2026 that no NGC tag bridges both yet
  ([forum thread](https://forums.developer.nvidia.com/t/vllm-26-04-py3-compatibility-mode-is-unavailable-on-driver-580-173-02-no-dgx-dashboard-path-to-595-58/379168)).
  When the DGX Dashboard ships a ≥ 595 driver, an NGC tag becomes an option: set `VLLM_IMAGE` in
  the env file (the script keeps your choice) and restart the teacher.
- The `vllm/vllm-openai` entrypoint is `vllm serve`; NGC images have none. The unit uses
  `--entrypoint vllm … serve MODEL`, which works with both.

### Teacher model: `google/gemma-4-31B-it`

- [Model card](https://huggingface.co/google/gemma-4-31B-it): exact id **`google/gemma-4-31B-it`**
  (capital B; the API returns an error for `gemma-4-31b-it`, and vLLM serves the name exactly as
  given, so the old lowercase value in the example env was wrong; bootstrap fixes it). Apache-2.0,
  **not gated**, revision `842da37`, `Gemma4ForConditionalGeneration`, BF16, 31.27 B parameters,
  **62.5 GB** of safetensors, 256K context. The card's serve command is plain
  `vllm serve "google/gemma-4-31B-it"`; recommended sampling `temperature=1.0, top_p=0.95, top_k=64`.
- Memory on the 128 GB Spark (unified CPU+GPU memory; `--gpu-memory-utilization` is a fraction of all of it):

  | | Estimate |
  |---|---|
  | Weights (BF16) | 62.5 GB |
  | KV cache per 8K-token sequence | ≈ 1.5 GB: 10 full-attention layers × 4 KV heads × 512 dim plus 50 sliding-window layers (window 1024) × 16 KV heads × 256 dim, BF16, from `config.json` |
  | `VLLM_GPU_UTIL=0.70` | ≈ 90 GB for vLLM → ≈ 25 GB for KV cache and activations (~15 full 8K sequences); ≈ 38 GB left for the OS, worker and small jobs |
  | old value `0.55` | ≈ 70 GB → only a few GB after weights; likely to fail to start or allow 1–2 sequences |

  These are estimates from the config. vLLM prints the real KV-cache size at startup
  (`journalctl --user -u vllm-teacher | grep -i 'KV cache'`). Stop the teacher before large
  training runs.
- Speed: decoding is memory-bandwidth bound (~273 GB/s), so expect roughly 4 tokens/s per
  sequence in BF16; Spark users report 3–7 tok/s single-stream
  ([forum](https://forums.developer.nvidia.com/t/slow-inference-with-31b-model-gemma-4-optimizations/366024)).
  Data generation should send many requests at once. To go about 2× faster, NVIDIA's
  [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) (32.6 GB, NVIDIA
  license, also on the playbook's list) is one `TEACHER_MODEL` change away. Check its license
  before you use its outputs in a public dataset.

### Training image: Unsloth

- NVIDIA's **DGX Spark Unsloth playbook** ([build.nvidia.com/spark/unsloth](https://build.nvidia.com/spark/unsloth),
  [source](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/unsloth), "Last Updated
  12/15/2025") uses `nvcr.io/nvidia/pytorch:25.11-py3` (CUDA 13.0) with
  `pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"` and
  `pip install --no-deps unsloth unsloth_zoo bitsandbytes`. The `--no-deps` matters: it stops pip
  from replacing the container's PyTorch and CUDA builds. Bootstrap only pulls the base image; each project's
  training job installs its pinned Unsloth on top of it.
- Unsloth also documents its own DGX Spark Dockerfile route
  ([Unsloth blog](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth)).
- Newer `nvcr.io/nvidia/pytorch` tags (up to `26.10-py3`) exist, but like the vLLM ones they likely need
  the newer driver; stay on the playbook's tag until the driver moves.

### Hugging Face cache and token

- Cache: `~/.cache/huggingface` on the host (`HF_HOME` default; models under `hub/`), see
  [cache guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) and
  [environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).
  The teacher mounts it at `/root/.cache/huggingface`, the playbook's layout. Bootstrap downloads
  as your uid (`HF_HOME=/hf` in the container), so the cache stays writable for your own jobs.
- Token: a **read** token ([tokens docs](https://huggingface.co/docs/hub/en/security-tokens)) in
  `HF_TOKEN` in `~/.config/spark-lab.env` (mode 600). It is passed to containers with
  `-e HF_TOKEN` and never shows up on a command line; bootstrap checks it by sending it to
  `whoami-v2` on stdin. Gemma 4 isn't gated, so the token only avoids anonymous rate limits. The
  write token stays with `publish/` (rule 3).
- vLLM's OpenAI-compatible server: [docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
  The unit binds it to `127.0.0.1:8000` only.
- Phone alerts: [ntfy](https://docs.ntfy.sh/). Anyone who knows the topic can read your alerts, so bootstrap offers a random one.

## Troubleshooting

- **Teacher fails with not enough free memory at startup.** On unified memory, page cache counts
  as used. Run `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` (from the playbooks), or
  lower `VLLM_GPU_UTIL`, then `systemctl --user restart vllm-teacher`.
- **`permission denied` on the docker socket from the services.** The user's systemd manager
  started before you joined the docker group: `sudo systemctl restart user@$(id -u).service`.
  Bootstrap checks for this.
- **Smoke test answers in English or Urdu script.** The teacher is up, but it isn't following the system
  prompt. Check that `TEACHER_MODEL` is the instruction-tuned `-it` model.
