# Log classifier iteration history

Each change below was made after looking at `heldout_test`, so that split is no longer a
perfectly clean test set; the history is here so nobody mistakes the last number for a
first-try result. A fresh test set (new fault variants written after this, or real GPU
faults from the Spark) is the honest check.

| Version | Change | heldout_test model acc | acts on (threshold) | precision when acting |
| --- | --- | --- | --- | --- |
| v1 | word uni/bigrams, C picked on val + heldout_val | 0.254 | 10.2% | 0.526 |
| v2 | split CamelCase/snake_case, extra features for the error lines | 0.355 | 17.5% | 0.337 |
| v3 | regex rule hits as features, C by leave-one-variant-out CV | 0.418 | 30.4% | 0.565 |
| v4a | 28 more training variants (75 total; held-out variants unchanged), 60 seeds each | 0.636 | 50.0% | 0.714 |
| v4 | v4a retrained after the error-line regex also matched bare `NotImplementedError` lines (needed by the no-evidence guard) | 0.683 | 50.5% | 0.717 |

v4 is the shipped model (`lab/models/log_classifier.json`). Tuning stopped here. On
`heldout_test` every wrong action (60 of 420 rows) comes from one variant,
`disk_full.sqlite` ("database or disk is full"), read as `flaky` because the training
`flaky.sqlite_locked` variant says "database is locked". The cost is bounded: `flaky`
gets at most two plain retries, then an incident. The other six unseen variants are
either right or escalated (escalation is the designed behaviour when unsure).

The regex rules were written before the chaos catalogue and never updated for the v4
variants, which is why their in-distribution accuracy fell from 0.73 to 0.60. They score
1.000 on `heldout_val` because those error strings are common real ones the rules list.
