# Held-out-variant mistakes and escalations

## bad_data.pickle -> unknown (0.39 code_bug)

```
           ~~~~~~~~~~~^^^^
_pickle.UnpicklingError: invalid load key, '<'.
=== exit 1
```

## code_bug.unbound -> unknown (0.53 code_bug)

```
           ^^^^
UnboundLocalError: cannot access local variable 'loss' where it is not associated with a value
=== exit 1
```

## cuda_error.cudnn -> unknown (0.35 code_bug)

```
    raise RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
=== exit 1
```

## disk_full.sqlite -> flaky (0.68 flaky)

```
    raise sqlite3.OperationalError("database or disk is full")
sqlite3.OperationalError: database or disk is full
=== exit 1
```

## flaky.eintr -> unknown (0.55 flaky)

```
    raise InterruptedError(errno.EINTR, "Interrupted system call")
InterruptedError: [Errno 4] Interrupted system call
=== exit 1
```

## network.reset -> unknown (0.45 network)

```
    raise RemoteProtocolError("Server disconnected without sending a response.")
httpx.RemoteProtocolError: Server disconnected without sending a response.
=== exit 1
```

## oom.vllm_kv -> unknown (0.29 oom)

```
                     "when initializing the engine.")
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
=== exit 1
```

## Counts

- bad_data.pickle -> unknown: 51
- code_bug.unbound -> unknown: 2
- cuda_error.cudnn -> unknown: 60
- disk_full.sqlite -> flaky: 60
- flaky.eintr -> unknown: 26
- network.reset -> unknown: 9
- oom.vllm_kv -> unknown: 60
