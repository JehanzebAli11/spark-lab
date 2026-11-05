# Log classifier results

TF-IDF (4630 features) + logistic regression, C=30 (leave-one-variant-out CV), confidence threshold 0.550 (lowest with >= 98% precision on val + heldout_val).
Rows below the threshold are escalated (incident file) instead of auto-fixed.

| Split | n | Rules acc | Model acc | Model macro-F1 | +threshold: acts on | precision when acting | wrong actions | hybrid: acts on | hybrid precision | hybrid wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2562 | 0.604 | 1.000 | 1.000 | 100.0% | 1.000 | 0 | 100.0% | 0.927 | 187 |
| val | 549 | 0.601 | 1.000 | 1.000 | 100.0% | 1.000 | 0 | 100.0% | 0.865 | 74 |
| test | 549 | 0.605 | 1.000 | 1.000 | 100.0% | 1.000 | 0 | 100.0% | 0.933 | 37 |
| heldout_val | 420 | 1.000 | 0.729 | 0.648 | 61.4% | 0.950 | 13 | 100.0% | 1.000 | 0 |
| heldout_test | 420 | 0.714 | 0.683 | 0.604 | 50.5% | 0.717 | 60 | 87.9% | 0.813 | 69 |

Serving mode picked on val + heldout_val: **model**.

Rules accuracy counts a rule miss (`unknown`) as wrong. `heldout_test` uses fault variants the model never saw, one per class.

Per-class F1 on heldout_test (model, no threshold):

| Class | Precision | Recall | F1 | n |
| --- | --- | --- | --- | --- |
| oom | 1.000 | 1.000 | 1.000 | 60 |
| disk_full | 0.000 | 0.000 | 0.000 | 60 |
| cuda_error | 1.000 | 0.050 | 0.095 | 60 |
| network | 1.000 | 1.000 | 1.000 | 60 |
| bad_data | 1.000 | 0.733 | 0.846 | 60 |
| flaky | 0.500 | 1.000 | 0.667 | 60 |
| code_bug | 0.451 | 1.000 | 0.622 | 60 |
