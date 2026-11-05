# Incident: cricket-pose-train

- **For:** owner
- **Class:** cuda_error (confidence 1.00, model)
- **Why stopped:** cuda_error fix already tried 1 time(s)
- **Exit code:** 1 after 2 attempt(s)
- **Log:** `logs/014-20261105T072355055506-cricket-pose-train.log`
- **Evidence:** `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`

## Next steps

The CUDA error came back after a cool-down. Check `nvidia-smi -q`, dmesg for Xid lines, and whether the same step fails every time (that suggests a code bug, not hardware).

## Healing so far

- attempt 1: cuda_error (0.99) -> cooldown_retry: waited 60 s
- attempt 2: cuda_error (0.99) -> incident: cuda_error fix already tried 1 time(s)

## Log tail

```
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

=== exit 1
=== heal: cuda_error (1.00, model) -> cooldown_retry: waited 60 s

=== 2026-11-05T07:23:56+00:00 attempt 2 :: /usr/local/bin/python -m lab.chaos --fault cuda_error.illegal_access --seed 4 --fail-attempts 5
python 3.13.18 | job cricket-pose-train | attempt 2 | batch_size 64
Epoch 0:   1%|          | 2/195 [00:02<03:00, 4.13it/s] loss=2.527 v_num=4
Epoch 0:   2%|          | 4/195 [00:04<03:00, 2.41it/s] loss=2.518 v_num=5
WARNING: CUDA initialization: the cuda driver reported version 13.0 (ok)
Epoch 0:   3%|          | 6/195 [00:06<03:00, 4.29it/s] loss=2.534 v_num=8
Epoch 0:   4%|          | 8/195 [00:08<03:00, 2.26it/s] loss=2.473 v_num=8
Epoch 0:   5%|▍         | 10/195 [00:10<03:00, 2.32it/s] loss=2.424 v_num=4
Epoch 0:   6%|▍         | 12/195 [00:12<03:00, 2.95it/s] loss=2.449 v_num=9
Epoch 0:   7%|▍         | 14/195 [00:14<03:00, 2.01it/s] loss=2.419 v_num=4
Epoch 0:   8%|▍         | 16/195 [00:16<02:00, 1.04it/s] loss=2.354 v_num=4
Epoch 0:   9%|▍         | 18/195 [00:18<02:00, 4.44it/s] loss=2.390 v_num=2
Epoch 0:  10%|█         | 20/195 [00:20<02:00, 2.95it/s] loss=2.322 v_num=3
Epoch 0:  11%|█         | 22/195 [00:22<02:00, 1.76it/s] loss=2.323 v_num=6
Epoch 0:  12%|█         | 24/195 [00:24<02:00, 3.70it/s] loss=2.287 v_num=5
disk usage 82% on /home/lab (41 GB free)
Epoch 0:  13%|█         | 26/195 [00:26<02:00, 1.99it/s] loss=2.238 v_num=9
Epoch 0:  14%|█         | 28/195 [00:28<02:00, 2.52it/s] loss=2.233 v_num=2
Epoch 0:  15%|█▍        | 30/195 [00:30<02:00, 2.66it/s] loss=2.190 v_num=1
Epoch 0:  16%|█▍        | 32/195 [00:32<02:00, 4.62it/s] loss=2.260 v_num=5
retrying request to http://localhost:8000/v1 after connection reset (attempt 1/5) ... ok
Epoch 0:  17%|█▍        | 34/195 [00:34<02:00, 5.82it/s] loss=2.171 v_num=6
Epoch 0:  18%|█▍        | 36/195 [00:36<02:00, 5.32it/s] loss=2.207 v_num=1
Epoch 0:  19%|█▍        | 38/195 [00:38<02:00, 1.80it/s] loss=2.128 v_num=7
Epoch 0:  20%|██        | 40/195 [00:40<02:00, 2.10it/s] loss=2.107 v_num=9
Epoch 0:  21%|██        | 42/195 [00:42<02:00, 2.52it/s] loss=2.153 v_num=0
Epoch 0:  22%|██        | 44/195 [00:44<02:00, 3.16it/s] loss=2.072 v_num=5
Epoch 0:  23%|██        | 46/195 [00:46<02:00, 2.69it/s] loss=2.093 v_num=4
Epoch 0:  24%|██        | 48/195 [00:48<02:00, 3.08it/s] loss=2.051 v_num=3
Epoch 0:  25%|██▍       | 50/195 [00:50<02:00, 1.88it/s] loss=2.016 v_num=0
Saving model checkpoint to runs/chaos/cricket-pose-train/checkpoint-50
Epoch 0:  26%|██▍       | 52/195 [00:52<02:00, 3.90it/s] loss=2.027 v_num=2
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "~/lab/spark-lab/lab/chaos.py", line 788, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "~/lab/spark-lab/lab/chaos.py", line 782, in main
    _load_fault(a.fault)(ctx)
    ~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/lab/.venv/lib/python3.12/site-packages/torch/_tensor.py", line 7, in run
    backward(None)
    ~~~~~~~~^^^^^^
  File "/home/lab/.venv/lib/python3.12/site-packages/torch/_tensor.py", line 5, in backward
    raise AcceleratorError("CUDA error: an illegal memory access was encountered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n")
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

=== exit 1
=== heal: cuda_error (1.00, model) -> incident: cuda_error fix already tried 1 time(s)
```

## Job

```json
{
  "name": "cricket-pose-train",
  "cmd": [
    "/usr/local/bin/python",
    "-m",
    "lab.chaos",
    "--fault",
    "cuda_error.illegal_access",
    "--seed",
    "4",
    "--fail-attempts",
    "5"
  ],
  "cwd": ".",
  "priority": 14,
  "timeout_min": 2,
  "retries": 1,
  "resume_dir": null,
  "env": {
    "PYTHONPATH": "~/lab/spark-lab"
  },
  "batch_env": null,
  "report": null
}
```
