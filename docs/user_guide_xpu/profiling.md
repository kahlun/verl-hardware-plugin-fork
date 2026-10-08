# Intel XPU Profiling Guide

Last updated: 10/06/2026.

This guide describes the default profiling support adapted in `verl-hardware-plugin` for Intel XPU. The default path reuses verl community `global_profiler.tool=torch` and adds XPU activity support via verl-core's platform profiler hooks (`PlatformBase.torch_profiler_activity()` / `torch_profiler_content_name()`).

## Prerequisites

Install both `verl` and `verl-hardware-plugin` in editable mode, then enable the plugin in Ray runtime:

```yaml
working_dir: ./
excludes: ["/.git/"]
env_vars:
  VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
```

`PlatformXPU` overrides the two profiler hooks when `verl_hardware_plugin` is imported and the XPU platform is active

## Community Torch Profile

`PlatformXPU.torch_profiler_activity()` returns `torch.profiler.ProfilerActivity.XPU`, and `torch_profiler_content_name()` returns `"xpu"`. verl-core's `TorchProfilerToolConfig` and `get_torch_profiler` accept `xpu` in `contents` and collect that activity whenever both hooks are set on the active platform.

Example Hydra overrides:

```bash
python -m verl.trainer.main_ppo \
  ... \
  global_profiler.tool=torch \
  global_profiler.steps=[2] \
  global_profiler.save_path=outputs/profile \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=False \
  actor_rollout_ref.actor.profiler.ranks=[0] \
  actor_rollout_ref.actor.profiler.tool_config.torch.contents=[xpu,cpu,memory,shapes,stack] \
  actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True
```

Common `contents` values:

| Value | Description |
| --- | --- |
| `xpu` | Collect XPU device activities. |
| `cpu` | Collect CPU activities. |
| `memory` | Enable memory profiling. |
| `shapes` | Record operator input shapes. |
| `stack` | Record Python stack traces. |

`xpu` and `cuda` cannot both be recorded in the same run -- only one device activity is collected per profiling session. If `contents` names both, `xpu` wins (there is no CUDA device on an XPU platform) and verl-core logs a warning that `cuda` was ignored.

Outputs are written as Chrome trace files under `global_profiler.save_path`, for example:

```text
outputs/profile/e2e/prof_rank-0_<pid>_<timestamp>.json.gz
```

Open the trace with Chrome `chrome://tracing` or any compatible trace viewer.

## Profiling Steps and Scope

Use `global_profiler.steps` to select training steps. Example: `global_profiler.steps=[2,3]`.

Set `global_profiler.profile_continuous_steps=True` to combine continuous steps into one profiling window. Keep it `False` to generate one output window per selected step.

Enable profiling only on the worker roles you need. For example, use `actor_rollout_ref.actor.profiler.enable=True` for actor paths, and enable `actor_rollout_ref.ref.profiler.enable=True` or `critic.profiler.enable=True` only when those roles are part of the job.

## Troubleshooting

- If `xpu` is rejected in `contents`, confirm `VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"` is set in Ray `runtime_env.yaml`, and that verl-core includes the platform profiler hooks (verl-project/verl#7988 or later).
- If no files are generated, check that `global_profiler.steps` matches the actual `global_steps` printed by training.
- Profiling can add significant overhead; start with one step and rank 0, then expand scope only when needed.
