# HANDOFF: self-healing queue, chaos log dataset, training-run doctor

Branch `self-healing`. Written by a cloud session (CPU only, no GPU, no Hugging Face, no API
keys) for the owner to continue with Claude Code on the DGX Spark. Updated after every
milestone; if this file looks unfinished, the session ran out of credits mid-way and the
status table is the source of truth.

## 1. Status

| Part | State | Evidence |
| --- | --- | --- |
| Self-healing queue (classify failures, apply fixes, incidents, report) | done, 35 tests pass | `tests/test_heal.py`, `results/demo/queue_report.md` |
| Chaos jobs + labelled log dataset | done (4,500 rows, 75 variants) | `datasets/chaos-logs/` |
| Log classifier (CPU) + confidence threshold | done, v4 shipped | `results/log_classifier/report.md`, `ITERATIONS.md` |
| Training-run doctor (sim runs, image + features models, auto-fix loop) | dataset done (4,410 runs); models training | `datasets/doctor-runs/` |
| Dataset cards, figures, demo GIF, README | not started | |

## Stretch list

(kept up to date as the core lands)
