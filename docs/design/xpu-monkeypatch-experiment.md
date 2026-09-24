# Experiment: how much of #7917 can live in the plugin as pure monkeypatch?

## Context

`verl-hardware-plugin#26` asked maintainers whether Intel XPU should extend
`PlatformBase` in `verl-core` (the TPU precedent) or stay entirely inside
this plugin via monkeypatch (the MLU precedent). The maintainer's answer
(2026-09-23): default to the plugin, keep `verl-project/verl#7917` (which
adds `is_reduce_avg_supported`/`attention_utils_module`/`profiler_markers`/
`dist_profiler_cls` to `PlatformBase`) narrowed to only what truly can't be
done from here.

This branch is the answer to "then how much of #7917 actually can't be done
from here" — implemented, not just argued. It does not touch
`feature/xpu-vtune-avg-attention` (the branch behind open PR #22, which
still uses the hook-based approach and depends on #7917 merging). This is a
separate, from-`main` branch so the two can be compared directly for size
and how much they depend on core changes ever landing.

## Verdict per capability

| Capability | In this branch? | Mechanism | Notes |
|---|---|---|---|
| `attention_utils_module` | Yes | Patch `verl.utils.attention_utils._get_attention_functions` | Clean. That function is re-invoked on every call by the public wrappers, so patch order doesn't matter. |
| `is_reduce_avg_supported` (FSDP2) | Yes | Not a patch at all — extended `verl_hardware_plugin/engines/fsdp_xpu.py`'s existing `initialize()` to loop over every FSDP2-wrapped submodule, not just the root | See below — this is *better* than the monkeypatch this branch originally planned. |
| `is_reduce_avg_supported` (3 raw `dist.all_reduce(op=AVG)` sites) | Yes, flagged | Patch `torch.distributed.all_reduce` process-wide when `op=AVG` | Robust to import order (attribute lookup at call time), but changes global `ReduceOp.AVG` semantics for the whole process. Real design smell, documented in `patches/reduce_avg_allreduce_patch_xpu.py`. |
| `dist_profiler_cls` (VTune tool selection) | Yes | Patch `DistProfiler.__init__` (mutate the class method, not the name) | Clean — see `patches/dist_profiler_patch_xpu.py` docstring for why patching the method and not reassigning the class name matters. |
| `profiler_markers` (ambient `marked_timer`/`mark_start_range` calls throughout the trainer) | **No** | — | Genuine blocker. `verl/utils/profiler/__init__.py` picks one of three whole modules with an `if/elif/else` **once**, at whatever moment it's first imported by *anything* in the process — there's no per-call dispatch object to patch. Reassigning `verl.utils.profiler.marked_timer` after the fact only helps callers that read that name *after* the patch runs; anything that already did `from verl.utils.profiler import marked_timer` (or `from verl.utils.debug import *`, which re-exports it) earlier keeps its own bound reference. Whether this patch would work in practice depends on winning a race against verl-core's own import order — not a foundation to build on. |
| `dist_profiler_cls` — rollout-server profiler allowlist (vLLM/SGLang/TRT-LLM async servers) | **No** | — | Genuine blocker, same conclusion as `verl-hardware-plugin#26`: `profiler_config = None` and the call that consumes it are in the same method body, with no hook or registry in between. Nothing to intercept short of forking ~40 lines of server constructor per engine. |
| `set_numa_affinity` (pyzes NUMA pinning — **not** part of #7917) | Yes | Patch `verl.utils.distributed.set_numa_affinity` wholesale | Clean, but only because `apply_all()` runs before either call site is imported — and it *must not* be applied from `PlatformXPU.__init__`. See below. Replaces the core-hook approach in `verl-hardware-plugin-fork#9` + `kahlun/verl#19`. |

Net: 3 of 4 hooks from #7917 are unnecessary — this plugin already does what
they'd do, with zero core changes. The 4th (`profiler_markers`) turns out to
have the *same* import-order problem as the rollout-server case, which
earlier framing (including an earlier version of this repo's own proposal
doc) had marked as "clean." It isn't. Two things, not one, look like
legitimate small-core-PR asks now — not the 4 hooks #7917 currently has.

## Why the FSDP2 fix isn't `patches/reduce_avg_fsdp2_patch_xpu.py`

The first draft of this plan (matching the proposal doc) was to wrap
`verl.utils.fsdp_utils.apply_fsdp2`: call the original, then walk
`model.modules()` and call `set_force_sum_reduction_for_comms(True)`.

That doesn't actually work reliably: `verl/workers/engine/fsdp/
transformer_impl.py` does `from verl.utils.fsdp_utils import apply_fsdp2` at
its own import time. Reassigning `verl.utils.fsdp_utils.apply_fsdp2` after
that has already happened has no effect on the name bound inside
`transformer_impl.py` — the exact same class of import-order fragility as
`profiler_markers` above.

This plugin doesn't need that patch at all, though: `fsdp_xpu.py`'s engine
subclasses already call `set_force_sum_reduction_for_comms` on the
constructed model *after* `super().initialize()` returns — from inside their
own `EngineRegistry`-registered class, which verl-core calls directly (no
patching involved). This branch just extended that existing, already-real
mechanism to loop over every FSDP2-wrapped submodule instead of only the
root (`_force_sum_reduction_on_all_fsdp_modules` in `fsdp_xpu.py`), which is
what `is_reduce_avg_supported`'s FSDP2 wiring in #7917 would have bought —
achieved here with no patch and no hook.

## Why `set_numa_affinity` can't be patched from `PlatformXPU.__init__`

Every other patch in this package is applied from `PlatformXPU.__init__`, on
the deliberate reasoning in `patches/__init__.py`: only patch once verl has
actually *selected* XPU, so a mixed host with `VERL_PLATFORM=nvidia` doesn't
get its semantics changed just because Intel GPUs happen to be present.

That reasoning breaks for this one capability, because of the order of two
adjacent lines in `verl/model_merger/megatron_model_merger.py`:

```python
set_numa_affinity()  # line 154
torch.distributed.init_process_group(get_nccl_backend())  # line 155
```

`get_nccl_backend()` is what would first call `get_platform()` and therefore
first construct `PlatformXPU` — one line *after* the function we need
patched has already run. `BaseModelMerger.__init__` never touches the
platform either. So on the checkpoint-merge path a patch applied from
`PlatformXPU.__init__` would never have been installed, and
`set_numa_affinity()` would silently no-op: the exact failure shape as the
bug being fixed. (The training path is fine either way —
`engine_workers.py:90` calls `initialize_global_process_group_ray()`, which
goes through `get_device_name()`/`get_nccl_backend()`, before line 92.)

So this patch is applied from `apply_all()` at plugin-import time instead,
and the "is XPU actually selected?" check moves from *apply* time to *call*
time (`_platform_is_xpu()`). That preserves the property the original
reasoning was protecting — non-XPU platforms keep their exact pynvml path —
without depending on the platform singleton already existing. It also can't
be done the other way round: calling `get_platform()` during plugin import
would freeze the singleton before anything could call `set_platform()`.

Import order is what makes the patch itself land, and that part is
guaranteed rather than lucky: `verl/__init__.py` loads plugin entry_points
after importing only `.protocol`, `.utils.device`, `.utils.import_utils` and
`.utils.logging_utils` — none of which reach `engine_workers` or
`model_merger` — and importing any `verl.*` submodule always runs
`verl/__init__.py` first. `_rebind_importers()` covers the two from-import
call sites anyway, so that if verl-core's import order ever changes this
degrades to a working patch instead of a silent no-op.

## Comparison

| | Files touched outside this plugin | Depends on #7917 merging |
|---|---|---|
| PR #22 (`feature/xpu-vtune-avg-attention`) | 0 (but 4 `PlatformXPU` hook methods are dead code without #7917) | Yes, for all 4 hooks |
| This branch | 0 | No, for 3 of 4. `profiler_markers`'s ambient markers and the rollout-server allowlist remain open — either fork ~40 lines of server code into the plugin, or ask for the two small core changes verl-hardware-plugin#26 already discusses. |

## Not yet done

- No hardware validation for the four #7917-related capabilities — those were
  built and reasoned through against `verl-core`'s real `main` source (via
  GitHub, not a live checkout), not run on a B60. Treat their "clean" verdicts
  above as "verified against source," not "verified end-to-end."
- Unit tests cover `numa_affinity_patch_xpu` only
  (`tests/test_numa_affinity_patch_xpu.py`, 24 cases, passing — verl and pyzes
  are both stubbed so it runs on a CPU-only host). `attention_patch_xpu`,
  `dist_profiler_patch_xpu` and `reduce_avg_allreduce_patch_xpu` still have
  none.
- `set_numa_affinity` is the exception — it **is** hardware-verified. See the
  section below.
- Watch item for a multi-GPU/Ray run: `_resolve_local_rank()` returns the
  global device id Ray assigned, while pyzes enumerates only the devices
  visible to the process. Those agree when
  `RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK` is set and can disagree when Ray
  sets `ZE_AFFINITY_MASK` per worker. The pynvml path this replaces has the
  identical property, so it isn't a regression introduced here — but a log
  line reading `out of range for N zes-visible device(s)` is this, and it
  would mean the same latent issue exists on CUDA today. The verification run
  below had `ZE_AFFINITY_MASK` unset, so it did not exercise the disagreeing
  case.

## Hardware verification: `set_numa_affinity` (2026-09-24)

Run via `devctl test --gpu=2 --gpu-model=B60` on image
`verl-intel-gpu:pr8-monkeypatch-20260924`, with this branch's
`verl_hardware_plugin` overlaid onto the image's editable install
(`scripts/devctl_verify_numa.sh` → `scripts/verify_numa_patch_xpu.py`).
6/6 checks passed, exit 0:

```
set_numa_affinity -> verl_hardware_plugin.patches.numa_affinity_patch_xpu.apply.<locals>._patched_set_numa_affinity
[PASS] verl.utils.distributed.set_numa_affinity is patched
[PASS] verl selected the XPU platform            device_name='xpu' vendor='intel'
[PASS] engine_workers' from-imported binding is the patched function
  local_rank=0 bdf=0000:3d:00.0 numa_node=0 cpulist=0-127,256-383
[PASS] resolved PCI BDF via pyzes
[PASS] affinity pinning changed this process's cpuset    512 cpus -> 256 cpus
```

Three things this establishes that source review could not:

1. The patch really is installed inside a real verl process, with **zero**
   verl-core changes — the plugin was loaded through its normal
   `verl.plugins` entry point, nothing else.
2. The import-order argument holds *live*: `verl.workers.engine_workers`'s
   own from-imported binding is the patched function, not the original
   pynvml one. This is the claim the whole approach rests on, and it is the
   one that a silent no-op would have hidden.
3. Pinning actually happens: the process's cpuset went from all 512 logical
   CPUs to the 256 local to the GPU's NUMA node. Before this patch, that
   call was a no-op on XPU.

### Found by this run: pyzes 0.1.1 has no PCI API

The first attempt failed with
`AttributeError: module 'pyzes' has no attribute 'zes_pci_properties_t'`.
The image ships **pyzes 0.1.1**, which contains *no* PCI symbols at all —
`zesDevicePciGetProperties`, `zes_pci_properties_t` and `zes_pci_address_t`
were all added by oneapi-src/level-zero#462 and first released in **0.1.2**
(2026-06-12). Confirmed by diffing `dir(pyzes)` across both versions.

This is a latent defect in the implementation inherited from
`verl-hardware-plugin-fork#9`, and it applies to the `PlatformBase`-hook
version of this code equally: the failure surfaces as an unhelpful
`AttributeError`, swallowed into a warning, leaving the run silently
un-pinned. `_zes_device_pci_bdf()` now checks for the symbol up front and
raises a message naming the required version instead.

Consequence for the image: `pyzes>=0.1.2` needs to land in
`docker/intel_gpu/requirements-intel-gpu.txt` (the `feature/xpu-docker`
branch). That file currently pins neither pyzes nor pynvml — the image gets
0.1.1 transitively.
