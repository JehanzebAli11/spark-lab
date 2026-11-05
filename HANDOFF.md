# HANDOFF: self-healing queue, chaos log dataset, training-run doctor

Branch `self-healing`. Written by a cloud session (CPU only, no GPU, no Hugging Face, no API
keys) for the owner to continue with Claude Code on the DGX Spark. Updated after every
milestone; if this file looks unfinished, the session ran out of credits mid-way and the
status table is the source of truth.

## 1. Status

| Part | State | Evidence |
| --- | --- | --- |
| Self-healing queue (classify failures, apply fixes, incidents, report) | in progress | |
| Chaos jobs + labelled log dataset | not started | |
| Log classifier (CPU) + confidence threshold | not started | |
| Training-run doctor (sim runs, image + features models, auto-fix loop) | not started | |
| Dataset cards, figures, demo GIF, README | not started | |

## Stretch list

(kept up to date as the core lands)
