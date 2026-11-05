# Auto-fix loop

Harmed runs from test + held-out task. Up to 3 rounds of diagnose -> change setting -> rerun. Rescued = the final rerun is no longer harmed against the same healthy reference.

| Policy | Harmed runs | Rescued | Rescue rate | Held-out task rescue rate | Reruns | Escalated | Healthy runs touched | Healthy runs hurt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 780 | 10 | 1.3% | 0.8% | 780 | 0 | 130/130 | 14/130 |
| doctor | 780 | 458 | 58.7% | 52.0% | 1551 | 35 | 8/130 | 3/130 |
| ensemble | 780 | 459 | 58.9% | 52.4% | 1585 | 33 | 10/130 | 3/130 |
| oracle | 780 | 536 | 68.7% | 66.0% | 1448 | 0 | 0/130 | 0/130 |

Rescued per cause:

| Cause | none | doctor | ensemble | oracle |
| --- | --- | --- | --- | --- |
| lr_too_high | 1/68 | 42/68 | 45/68 | 68/68 |
| lr_too_low | 1/127 | 67/127 | 66/127 | 119/127 |
| no_warmup | 4/29 | 20/29 | 20/29 | 8/29 |
| label_noise | 0/130 | 0/130 | 0/130 | 0/130 |
| overfitting | 2/75 | 58/75 | 59/75 | 61/75 |
| train_test_leak | 0/105 | 82/105 | 81/105 | 88/105 |
| dead_activations | 2/116 | 95/116 | 95/116 | 96/116 |
| exploding_gradients | 0/130 | 94/130 | 93/130 | 96/130 |
