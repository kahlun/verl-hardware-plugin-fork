# Intel VTune Profiling Guide

Last updated: 09/22/2026.

This guide describes Intel VTune (ITT) profiling support in `verl-hardware-plugin`
for Intel XPU. Unlike Cambricon MLU's profiling support (see
[`user_guide_mlu/profiling.md`](../user_guide_mlu/profiling.md)), which reuses
verl's built-in `global_profiler.tool=torch` path via a plugin-side monkey-patch,
VTune is wired in purely through the newer `PlatformBase` plugin hooks —
`PlatformXPU.profiler_markers()` and `PlatformXPU.dist_profiler_cls()` — which
verl core discovers without needing to know about ITT or XPU itself. `torch.profiler`
support for XPU is also possible (`torch.profiler.ProfilerActivity.XPU` exists
natively — more natively than MLU's own activity, which needs `torch_mlu` to
register it) — see "Torch Profiler" below; that path is proposed, not yet
implemented in verl-core.

## Prerequisites

- These hooks (`is_reduce_avg_supported`, `attention_utils_module`,
  `profiler_markers`, `dist_profiler_cls`) only exist on a verl-core build that
  includes [verl-project/verl#7917](https://github.com/verl-project/verl/pull/7917)
  ("enable intel XPU to Verl with plugin mechanism with extra General API
  abstraction"), currently **open, not yet merged**. Against stock verl-core
  `main`, `PlatformBase` has no such methods to override, so `profiler.tool=vtune`
  silently falls through to `DistProfiler`'s no-op fallback instead of erroring.
- Install both `verl` (from the `#7917` branch/ref) and `verl-hardware-plugin` in
  editable mode, and enable the plugin in Ray runtime:

  ```yaml
  working_dir: ./
  excludes: ["/.git/"]
  env_vars:
    VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
  ```

## Enabling VTune

There is no dedicated `tool_config.vtune` schema entry in verl-core's generated
config yet (only `nsys`/`npu`/`torch`/`torch_memory`/`precision_debugger` have
one) — `VtuneProfiler` reuses `NsightToolConfig`'s shape (it only reads
`tool_config.discrete`), so add it with Hydra's `+` (this exact recipe has been
verified end-to-end against a live Hydra compose, not just read from source):

```bash
python -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.actor.profiler.tool=vtune \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=False \
  actor_rollout_ref.actor.profiler.ranks=[0] \
  +actor_rollout_ref.actor.profiler.tool_config.vtune._target_=verl.utils.profiler.config.NsightToolConfig \
  +actor_rollout_ref.actor.profiler.tool_config.vtune.discrete=False
```

Omitting the `tool_config.vtune` block entirely raises
`AssertionError: tool_config must be provided when profiler is enabled` from
`VtuneProfiler.__init__` — this is intentional, not a bug to work around.

## How this differs from Nsight/torch/NPU profiling

Nsight, torch, and NPU profiling are process-level: `profiler.start()`/`stop()`
tell the backend when to begin and end recording, and each writes a self-contained
trace file under `global_profiler.save_path`. **VTune does not work this way.**
`PlatformXPU.profiler_start()`/`profiler_stop()` are no-ops by design — VTune
attaches externally as a collector and observes `range_push`/`range_pop` events
rather than being started/stopped by the profiled process itself. The real
signal is the ITT range markers (`mark_start_range`/`mark_end_range`, wrapping
each named stage — `compute_log_prob`, `update_actor`, etc. — via
`profiler_markers()`), which are only visible when the training process runs
*under* an actual VTune collector, e.g.:

```bash
vtune -collect hotspots -result-dir ./vtune_results -- \
  python -m verl.trainer.main_ppo ... actor_rollout_ref.actor.profiler.tool=vtune ...
```

No trace file appears under `global_profiler.save_path` for the `vtune` tool —
open `./vtune_results` in the VTune GUI/CLI instead. Because `profiler_start`/
`profiler_stop` are no-ops, `tool_config.discrete` has little practical effect
for `vtune` specifically (unlike `torch`/`nsys`, where it materially changes
what gets recorded) — the ITT ranges are emitted the same way either way.

## Torch Profiler (proposed — not yet implemented)

An alternative to VTune: a self-contained Chrome-trace file via verl's built-in
`torch` tool, the same one CUDA/MLU use, instead of an externally-attached
collector. This needs two new optional `PlatformBase` hooks
(`torch_profiler_activity()`/`torch_profiler_content_name()`) that do not exist
in verl-core yet — `PlatformXPU`'s side of it is implemented on this branch,
but it has no effect until the matching verl-core change lands (see the
companion proposal in `verl-core`'s `feature/torch-profiler-plugin-hook`
branch). Once both land, the recipe would be:

```bash
python -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.actor.profiler.tool=torch \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.tool_config.torch.contents=[xpu,cpu,memory,shapes,stack]
```

Unlike VTune, this writes a self-contained `.json.gz` Chrome trace under
`global_profiler.save_path` — no external collector needed, open directly in
`chrome://tracing` or Perfetto. Not yet verified against real hardware: whether
`torch.profiler.profile(activities=[CPU, XPU])` actually produces a populated,
correct trace on Intel Arc/oneAPI has not been tested.

## Troubleshooting

- `AssertionError: tool_config must be provided when profiler is enabled` — add
  the `+actor_rollout_ref.actor.profiler.tool_config.vtune.*` overrides shown
  above; this profiler has no config-group default the way `nsys`/`npu`/`torch` do.
- If `profiler.tool=vtune` appears to do nothing, confirm the running verl-core
  build actually includes #7917 (see Prerequisites) — against stock `main` this
  is a silent no-op, not an error.
- If no ITT ranges show up in VTune, confirm the process was launched *under* a
  VTune collector (`vtune -collect ... -- python ...`), not run standalone —
  standalone runs execute the same code but nothing is listening for the events.
