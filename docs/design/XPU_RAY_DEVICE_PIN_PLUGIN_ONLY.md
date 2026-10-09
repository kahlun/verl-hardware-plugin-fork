# Pinning each Ray worker's XPU without a verl-core change

This branch answers one question raised against
[verl-project/verl#8145](https://github.com/verl-project/verl/pull/8145): is the Ray
device-pinning fix *impossible* to do purely plugin-side?

**No. It is possible, and this branch does it.** What it costs is the argument for the core
hook — not impossibility.

The PR's section "A plugin cannot reach this code path in time" is too strong and should be
reworded. The sentence is true of the one method; it is false of the goal.

## The problem

`Worker._setup_env_cuda_visible_devices()` calls `set_device()` only when a
`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` variable is set. By default verl never pins and
relies on Ray's per-actor visibility mask relabeling devices so that index 0 is this rank's
device. torch-xpu discards `ZE_AFFINITY_MASK` once its lazy init runs, so every actor still
enumerates every device and `xpu:N` is physical device N regardless of the mask — so every rank
of the job runs on `xpu:0`. Nothing raises: the run dies out of memory, or collectives take
paths meant for ranks that share a device.

## The four routes, and what each costs

| Route | Works? | Cost |
| --- | --- | --- |
| **A.** Patch `Worker._setup_env_cuda_visible_devices` from `PlatformXPU.__init__` | Yes, today | Only in time because of an import-order accident that core never promised (below). Arriving late is silent. |
| **B.** Patch `Worker._configure_with_store` from `PlatformXPU.__init__` | Yes, always | Rewrites a private core store key, and depends on knowing that `__init__` reads `LOCAL_RANK` between the two methods. |
| **C.** Patch `Worker._setup_env_cuda_visible_devices` at plugin *import* time, from `verl_hardware_plugin/__init__.py` | Yes, always | Not gated on the platform verl selected: core's pinning decision is rewritten for whatever platform wins, on any host with this plugin installed. This is the pattern rejected in plugin PR #8. |
| **D.** No code: set `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` | Yes, today | Works only because that entry is *dead*: Ray moved its Intel GPU manager to `ZE_AFFINITY_MASK` in [ray#64440](https://github.com/ray-project/ray/pull/64440) and deleted that NOSET name, so Ray still masks the actor while verl's hardcoded list flips its gate and pins the physical id. Dies the day anyone prunes the list — which #8145's `ray_utils.py` hunk is about. |

`verl_hardware_plugin/patches/xpu/ray_device_pin_patch.py` on this branch implements **A**,
with **B** as an automatic fallback when A is installed too late. Read its module docstring:
it is written to be read by someone deciding between this and the core hook.

## Why route A is in time today — and why that is not a contract

Measured, not reasoned about (CPU-only host, verl core at `main` + this plugin):

```
verl_hardware_plugin/__init__.py:24          register_all_engines()
verl_hardware_plugin/engines/__init__.py:44  from verl_hardware_plugin.engines import fsdp_flagos
verl_hardware_plugin/engines/fsdp_flagos.py:14  from verl.workers.engine.base import EngineRegistry
verl/workers/engine/__init__.py:17           from .fsdp import FSDPEngine, ...
verl/workers/engine/fsdp/__init__.py:14      from .fsdp_turbo_impl import FSDPTurboEngineWithLMHead
verl/workers/engine/fsdp/fsdp_turbo_impl.py:20  from .transformer_impl import FSDPEngineWithLMHead
verl/workers/engine/fsdp/transformer_impl.py:89 device_name = get_device_name()     <-- module level
verl/utils/device.py:102                     return get_platform().device_name
verl/plugin/platform/platform_manager.py:140 get_platform()                        <-- singleton built here
```

So the platform singleton — and therefore `PlatformXPU.__init__`, and therefore `apply()` — is
built during `import verl`, before any `Worker` exists in the process. That is what makes a
plugin-side patch of the first statement of `Worker.__init__` possible at all.

Every link in that chain is incidental:

- the trigger is *this plugin* importing `verl.workers.engine.base` eagerly to register engines;
- the platform call is a module-level `device_name = get_device_name()` in a core engine file;
- neither is documented as load-order API, and making either lazy is an unremarkable refactor.

If it changes, `PlatformXPU()` is first constructed by the `get_resource_name()` call *inside*
`_setup_env_cuda_visible_devices` itself, and route A patches a function that is already
executing. The failure mode is not an exception — it is two ranks quietly sharing a device.
`apply()` detects this by walking the call stack for a `Worker.__init__` frame and switches to
route B; a plugin needing stack inspection to decide which private core method to patch is the
point.

## The hazard that bit this branch's own tests

`_detect_platform_name()` probes platforms by constructing each one, so `PlatformXPU.__init__`
runs on hosts where XPU is not selected — the patch is installed on core's `Worker` class on a
CPU-only machine with no XPU present at all. `test_import_verl_already_installs_the_patch_on_
any_host` asserts exactly that, and `apply()` therefore has to re-check the selected platform at
*call* time instead of patch time. For the AVG patch that hazard costs process-wide
`all_reduce` semantics; here it would silently redirect core's device-pinning decision for
whatever platform did win.

## What the core hook replaces

`PlatformBase.ray_device_index()` (#8145) is ~12 lines of core policy — *resolve the index from
Ray's own assignment* — plus a 4-line `PlatformXPU` override stating the hardware fact *our
runtime ignores its own mask* (branch `fix/xpu-ray-device-index`). No private core method, no
load-order dependency, no stack inspection, no second patch target, nothing installed on hosts
that did not select XPU — and it is the hook `PlatformTPU.ray_local_rank_override()` already
needs and cannot currently reach, which is why
[#8138](https://github.com/verl-project/verl/pull/8138) hardcodes `if device_name == "TPU"` in
`single_controller` instead.

## Reproducing

```bash
pytest tests/test_xpu_ray_device_pin_patch.py -q   # 8 passed, mock-only, no accelerator
pytest tests/ -q                                   # 118 passed
```

The test file covers: the unpatched bug (rank lands on device 0), the patch pinning the physical
id Ray assigned, CPU-only actors and in-process (non-Ray) `Worker` construction left alone, the
late-apply fallback, the late-apply *silence* when the fallback is removed, and the import-time
installation described above.

## Verified on hardware: 2× Intel Arc Pro B60, 2026-10-09

Both arms in one pod, on the same two GPUs, back to back. The image is
`verl-intel-gpu:kahlunte-pr23-xpu-docker-fix2` — the revision where this collision was first
measured, and whose verl-core still has the gate (`PlatformBase.ray_device_index exists False`,
`NOSET gate still in core worker.py True`), so the control is a real control and not a core
already carrying #8145. **verl-core is byte-identical across the arms**; between them only one
module is added to the installed plugin and `PlatformXPU.__init__` is wired to call its `apply()`.

Probe: per-rank device identity at four points, including through verl's own
`_setup_env_cuda_visible_devices()`, `initialize_global_process_group_ray()` and
`create_device_mesh()`, plus a 6 GB allocation on rank 0 cross-checked against every rank's free
memory, plus SUM/AVG on WORLD and on the FSDP mesh. No `RAY_EXPERIMENTAL_NOSET_*` variable set in
either arm.

| | ARM A — plugin as shipped | ARM B — plugin + `ray_device_pin_patch` |
| --- | --- | --- |
| `Worker._setup_env_cuda_visible_devices` owned by | `verl.single_controller.base.worker` | `verl_hardware_plugin.patches.xpu.ray_device_pin_patch` |
| `LOCAL_RANK` after verl's setup | `<unset>`, `<unset>` | `0`, `1` |
| distinct physical GPUs, at actor start | 1 of 2 | 1 of 2 |
| distinct physical GPUs, after verl setup / init_pg / device_mesh | **1 of 2** (`4e00`, `4e00`) | **2 of 2** (`4e00`, `ba00`) |
| rank 0 allocates 6 GB | rank 1's free memory drops too, 23561 → 17351 → **SHARED** | rank 1 unchanged at 23908 → **SEPARATE** |
| all_reduce SUM, WORLD and `mesh[fsdp]` | OK | OK |
| all_reduce AVG, WORLD and `mesh[fsdp]` | 4 of 4 abort (oneCCL scheduler path) | 4 of 4 OK |

So route A does fix the real defect on real hardware, with no verl-core change: the patch is
installed from `PlatformXPU.__init__` during `import verl`, wins the race against
`Worker.__init__`, and pins the physical id Ray assigned. The fallback to route B never fired
(no warning, and `LOCAL_RANK` is already correct at step 2) — route A was in time, exactly as
the import chain above predicts, on this core revision.

One reporting artifact: ARM B's facts line `NOSET gate still in core worker.py False` is a
measurement of `inspect.getsource(Worker._setup_env_cuda_visible_devices)`, which after patching
returns the *wrapper's* source. Core's file is untouched; the gate is still there, with the
original method called inside the wrapper.

The AVG rows are the collision's second-order effect, not a separate finding: with both ranks on
one device oneCCL takes the scheduler path, which does not implement AVG (MLSL-4062). They flip
to OK for the same reason the UUIDs become distinct.
