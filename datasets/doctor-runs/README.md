# Dataset card: doctor-runs

4,410 tiny CPU training runs, each with one deliberately bad setting (or none), its logged
curves, its final metrics, a healthy reference run on the same data, and a picture of its
curves. Built to train and test the training-run doctor (`labml/doctor/`).

## Summary

| | |
| --- | --- |
| Files | `runs.jsonl.gz` (4,410 rows, ~9 MB), `images.npz` (4,410 x 64 x 64 x 3 uint8, ~6 MB) |
| Causes | `healthy`, `lr_too_high`, `lr_too_low`, `no_warmup`, `label_noise`, `overfitting`, `train_test_leak`, `dead_activations`, `exploding_gradients` |
| Runs per cause | 450 on four task families + 40 on a held-out family |
| Label source | code: the setting that was injected (`labml/doctor/sim.py`). No model labelled anything. |
| Compute | 16 min on 3 CPU cores (each run trains twice: the run and its reference) |
| Licence | MIT |

## How it was made

```bash
python -m labml.doctor.dataset --per-cause 450 --heldout-per-cause 40 --procs 3
```

Each run trains a numpy MLP (1-10 hidden layers, 32-256 wide, ReLU or tanh, Adam or SGD
with momentum, 400 steps, batch 64, cosine schedule) on a synthetic classification task
(`blobs`, `spiral`, `teacher`, `xor`; `rings` held out) and logs 50 points of train/val
loss and accuracy, gradient norm, learning rate and the share of dead (ReLU) or saturated
(tanh) hidden units. How each cause is injected is in the `sim.py` docstring. The images
are rendered by `labml/doctor/render.py`; full-size PNGs go to `data/doctor/images/`
(not committed).

## Harmed or not

An injected setting does not always hurt. Each run is compared with a healthy-recipe
reference on the same rows (`sim.reference_config`). `harmed` (rules in `dataset.py`):

| Cause | Harmed / runs |
| --- | --- |
| healthy | 0 / 490 (by definition) |
| lr_too_high | 253 / 490 |
| lr_too_low | 467 / 490 |
| no_warmup | 104 / 490 |
| label_noise | 490 / 490 (by definition: the labels are wrong) |
| overfitting | 241 / 490 (accuracy or over-confident test loss) |
| train_test_leak | 370 / 490 (val accuracy inflated > 3 points over clean test) |
| dead_activations | 433 / 490 |
| exploding_gradients | 490 / 490 |

The doctors train and test on healthy runs plus harmed runs only (3,338 rows); the 1,072
injected-but-unharmed runs show no symptom to diagnose. The overfitting loss rule would
also flag 66 of 490 healthy runs, so that label is the noisiest.

## Fields (`runs.jsonl.gz`)

| Field | Meaning |
| --- | --- |
| `id` | `<cause>-<task>-<seed>` |
| `cause`, `task`, `seed`, `split` | injected cause, task family, seed, split |
| `config` | every setting of the run (`sim.RunConfig`) |
| `series` | 50 logged points each of `train_loss`, `val_loss`, `train_acc`, `val_acc`, `grad_norm`, `lr`, `dead_frac`; `null` after divergence |
| `final` | `val_loss`, `val_acc`, `test_loss`, `test_acc` (clean test set), `diverged` |
| `reference` | the same for the healthy reference run |
| `harmed` | see above |
| `n_params`, `seconds` | model size, wall time for run + reference |

`images.npz` holds `images` (same order as the jsonl) and `ids`.

## Splits

`seed % 10` 0-6 train, 7 val, 8-9 test, on the four standard task families;
`heldout_task` = all `rings` runs (never seen in training).

## Limits

- Tiny MLPs on synthetic tabular tasks. Real LLM fine-tunes have different curve shapes
  (and far longer runs); the Spark plan in `HANDOFF.md` checks transfer on real runs.
- One cause per run. Real runs often have two.
- "No warmup" rarely hurts at this scale (104/490), and the warmup fix rarely helps; the
  harm is better fixed by lowering the learning rate (see `results/doctor/autofix.md`).
