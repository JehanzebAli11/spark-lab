# Training-run doctor results

Rows: healthy runs + injected runs that were actually harmed. Injected runs that came out fine (not used): {'lr_too_high': 237, 'lr_too_low': 23, 'no_warmup': 386, 'overfitting': 249, 'train_test_leak': 120, 'dead_activations': 57}.

| Doctor | Sees | Test acc | Test macro-F1 | Held-out task acc | Held-out task macro-F1 | Threshold | Test: acts on | Test: precision when acting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image | curve picture | 0.893 | 0.842 | 0.824 | 0.835 | 0.96 | 49.0% | 0.980 |
| curves | curve numbers | 0.910 | 0.866 | 0.886 | 0.845 | 0.97 | 64.2% | 0.965 |
| features | curves + telemetry | 0.984 | 0.983 | 0.969 | 0.972 | 0.20 | 100.0% | 0.984 |
| ensemble | image + features | 0.982 | 0.980 | 0.962 | 0.968 | 0.20 | 100.0% | 0.982 |

Per-cause F1 on test:

| Cause | image | curves | features | ensemble |
| --- | --- | --- | --- | --- |
| healthy | 0.863 | 0.899 | 0.966 | 0.966 |
| lr_too_high | 0.800 | 0.839 | 0.990 | 0.990 |
| lr_too_low | 0.929 | 0.938 | 1.000 | 1.000 |
| no_warmup | 0.353 | 0.471 | 1.000 | 0.978 |
| label_noise | 0.978 | 0.961 | 0.984 | 0.984 |
| overfitting | 0.955 | 0.919 | 0.955 | 0.955 |
| train_test_leak | 0.864 | 0.875 | 0.962 | 0.955 |
| dead_activations | 0.904 | 0.945 | 0.994 | 0.994 |
| exploding_gradients | 0.936 | 0.947 | 1.000 | 1.000 |

## Unseen causes (leave-one-cause-out, features doctor)

Trained without the cause, tested on its runs: does the doctor escalate or guess?

| Held-out cause | n | Escalated (confidence only) | Escalated (+ novelty) | Called healthy | Most common wrong labels |
| --- | --- | --- | --- | --- | --- |
| lr_too_high | 68 | 14.7% | 92.6% | 2.9% | exploding_gradients (3), healthy (2) |
| lr_too_low | 127 | 0.0% | 0.0% | 8.7% | label_noise (112), healthy (11) |
| no_warmup | 29 | 0.0% | 100.0% | 0.0% | - |
| label_noise | 130 | 16.2% | 16.2% | 71.5% | healthy (93), dead_activations (15) |
| overfitting | 75 | 0.0% | 28.0% | 0.0% | train_test_leak (54) |
| train_test_leak | 105 | 0.0% | 0.0% | 73.3% | healthy (77), overfitting (28) |
| dead_activations | 116 | 69.8% | 69.8% | 16.4% | healthy (19), label_noise (15) |
| exploding_gradients | 130 | 100.0% | 100.0% | 0.0% | - |

Features doctor with threshold + novelty: test acts on 97.4% at precision 0.983; held-out task acts on 97.9% at precision 0.968.

Rows per split: train: 2098; val: 330; test: 620; heldout_task: 290
