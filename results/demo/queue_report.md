# Nightly report 2026-11-05

**4 done, 4 failed, 0 still queued.**

## Jobs

| Job | Status | Attempts | Minutes | Resumed from | Log |
| --- | --- | --- | --- | --- | --- |
| cricket-pose-train | failed | 2 | 0 | - | `logs/014-20261105T072355055506-cricket-pose-train.log` |
| scam-radar-eval | failed | 1 | 0 | - | `logs/015-20261105T072355055614-scam-radar-eval.log` |
| sms-import | failed | 1 | 0 | - | `logs/016-20261105T072355055725-sms-import.log` |
| kv-cache-probe | failed | 2 | 0 | - | `logs/017-20261105T072355055858-kv-cache-probe.log` |
| scam-radar-sft | done | 3 | 0 | checkpoint-350 | `logs/010-20261105T072355054903-scam-radar-sft.log` |
| teacher-datagen | done | 3 | 0 | - | `logs/011-20261105T072355055155-teacher-datagen.log` |
| urdu-bench-shards | done | 2 | 0 | - | `logs/012-20261105T072355055287-urdu-bench-shards.log` |
| hf-cache-warm | done | 2 | 0 | - | `logs/013-20261105T072355055400-hf-cache-warm.log` |

## Self-healing

8 job(s) failed at least once: 4 recovered, 4 incident(s) written.

| Job | Attempt | Diagnosis | Confidence | Action | Detail |
| --- | --- | --- | --- | --- | --- |
| cricket-pose-train | 1 | cuda_error (model) | 0.99 | cooldown_retry | waited 60 s |
| cricket-pose-train | 2 | cuda_error (model) | 0.99 | incident | cuda_error fix already tried 1 time(s) |
| scam-radar-eval | 1 | code_bug (model) | 0.99 | incident | code_bug: retrying will not help |
| sms-import | 1 | bad_data (model) | 1.00 | incident | bad_data: retrying will not help |
| kv-cache-probe | 1 | unknown (model) | 0.43 | retry | unclassified; plain retry (1 left) |
| kv-cache-probe | 2 | unknown (model) | 0.43 | incident | unclassified and out of retries |
| scam-radar-sft | 1 | oom (model) | 0.99 | halve_batch | BATCH_SIZE 64->32, LAB_BATCH_SCALE=0.5, resume from checkpoint |
| scam-radar-sft | 2 | oom (model) | 0.99 | halve_batch | BATCH_SIZE 32->16, LAB_BATCH_SCALE=0.25, resume from checkpoint |
| teacher-datagen | 1 | network (model) | 1.00 | backoff_retry | waited 30 s |
| teacher-datagen | 2 | network (model) | 1.00 | backoff_retry | waited 60 s |
| urdu-bench-shards | 1 | disk_full (model) | 1.00 | prune_retry | pruned 14 old checkpoint(s), kept newest 2 |
| hf-cache-warm | 1 | flaky (model) | 1.00 | retry | - |

Incidents:

- `incidents/20261105-072356-cricket-pose-train.md` (cuda_error)
- `incidents/20261105-072356-scam-radar-eval.md` (code_bug)
- `incidents/20261105-072356-sms-import.md` (bad_data)
- `incidents/20261105-072357-kv-cache-probe.md` (unknown)

## Failures

- **cricket-pose-train** exited 1 after 2 attempt(s); see `logs/014-20261105T072355055506-cricket-pose-train.log`
- **scam-radar-eval** exited 1 after 1 attempt(s); see `logs/015-20261105T072355055614-scam-radar-eval.log`
- **sms-import** exited 1 after 1 attempt(s); see `logs/016-20261105T072355055725-sms-import.log`
- **kv-cache-probe** exited 1 after 2 attempt(s); see `logs/017-20261105T072355055858-kv-cache-probe.log`

## Machine

- Disk: 30 GB free of 271 GB
- GPU: (demo: CPU container)
- Next in queue: nothing
