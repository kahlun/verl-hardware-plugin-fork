# FAQ and Troubleshooting

## Why does `torch.xpu.is_available()` return `True` but verl still picks `nvidia`?

The plugin was installed but not discovered. Check:

- `pip show verl-hardware-plugin` succeeded and the environment you're
  running in is the one you installed into.
- `VERL_USE_EXTERNAL_PLUGINS` is not set to `none`.
- `VERL_PLATFORM=intel` as an explicit override if auto-detection order is
  the suspected issue (auto-detection depends on plugin registration order —
  deterministic, but order-dependent).

## Multi-GPU hangs or crashes with a memory/VA-space error

**Level Zero VA pressure (2-GPU+).** Level Zero maps Intel GPU device memory
into each process's CPU virtual address space (~2.2 TB `VmPeak` per
process). With many colocated Ray workers, this exhausts kernel page table
resources — the memory pressure is a driver artifact, not real RAM usage.
Mitigate with:

```bash
export RAY_memory_monitor_refresh_ms=0     # disables Ray's OOM monitor
export RAY_NUM_PRESTART_PYTHON_WORKERS=0   # reduces idle worker processes
```

## SGLang rollout fails on weight sync

`update_weights` over IPC fails on Intel GPU with a `ForkingPickler`
authentication error when crossing Ray actor boundaries. A POSIX
shared-memory workaround exists but has not been upstreamed. Use vLLM as the
rollout engine until this lands.

## SDPA / oneDNN crashes on startup

Do not manually propagate `ONEAPI_DEVICE_SELECTOR` to Ray workers. Invalid
values (for example `level_zero:`) can block oneDNN from finding its OpenCL
device and crash SDPA (oneDNN primitive init). For device placement, use
`ZE_AFFINITY_MASK` instead — the Docker image includes a startup guard that
repairs empty/invalid `ONEAPI_DEVICE_SELECTOR` values, but a bare-metal
environment will not have that guard.

## FSDP2 crashes or produces wrong loss/val_loss averages

XCCL (oneCCL) does not implement `ReduceOp.AVG` for `all_reduce`/
`reduce_scatter`. This plugin's `PlatformXPU.is_reduce_avg_supported()`
returns `False`, which makes verl core call
`model.set_force_sum_reduction_for_comms(True)` for FSDP2
(`verl/utils/fsdp_utils.py`) and route plain `all_reduce` averaging through
`verl.utils.device.all_reduce_avg()` (SUM + manual divide) instead of
`ReduceOp.AVG` directly. If you're seeing this on a fork that predates this
hook, you're missing the fix.

## Do I need `flash-attn`?

No — the CUDA `flash-attn` package only ships CUDA wheels and is not
installable on XPU. `attn_implementation=flash_attention_2` is confirmed
working on real B60 hardware via `transformers`' Hub kernel fallback
(`kernels-community/flash-attn2`), which requires `transformers>=5.17.0`
specifically (not verl's project-wide `transformers==5.9.0` pin — see
[`docker/intel_gpu/requirements-intel-gpu.txt`](../../docker/intel_gpu/requirements-intel-gpu.txt)
for why that divergence is safe).
