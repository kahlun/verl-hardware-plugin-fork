# Experiment: reduce_avg + attention support for XPU, pure monkeypatch

## Context

`verl-hardware-plugin#26` asked maintainers whether Intel XPU should extend
`PlatformBase` in `verl-core` (the TPU precedent) or stay entirely inside
this plugin via monkeypatch (the MLU precedent). The maintainer's answer
(2026-09-23): default to the plugin, keep `verl-project/verl#7917` (which
adds `is_reduce_avg_supported`/`attention_utils_module`/`profiler_markers`/
`dist_profiler_cls` to `PlatformBase`) narrowed to only what truly can't be
done from here.

This branch covers two of those four capabilities —
`is_reduce_avg_supported` and `attention_utils_module` — split out from the
original combined experiment (`experiment/xpu-monkeypatch-no-core-hooks`)
so this half, which has no unresolved design smell beyond the one flagged
below, can be reviewed and hardware-validated independently of the VTune
profiler half (`docs/design/xpu-monkeypatch-vtune-profiler.md`), which has
an open blocker.

Does not touch `feature/xpu-vtune-avg-attention` (the branch behind open
PR #22, which uses the hook-based approach and depends on #7917 merging).
Built from `main` so it can be compared directly for footprint and
core-change dependency.

## Verdict per capability

| Capability | Mechanism | Notes |
|---|---|---|
| `attention_utils_module` | Patch `verl.utils.attention_utils._get_attention_functions` | Clean. That function is re-invoked on every call by the public wrappers, so patch order doesn't matter. |
| `is_reduce_avg_supported` (FSDP2) | Not a patch at all — extended `verl_hardware_plugin/engines/fsdp_xpu.py`'s existing `initialize()` to loop over every FSDP2-wrapped submodule, not just the root | See below — this is *better* than the monkeypatch this branch originally planned, and matches the per-layer+root fix already validated on real B60 hardware via a 12-run controlled A/B (6/6 pass with the fix, 6/6 fail without, on `run_grpo_fsdp2_intel_gpu.sh`). |
| `is_reduce_avg_supported` (3 raw `dist.all_reduce(op=AVG)` sites) | Patch `torch.distributed.all_reduce` process-wide when `op=AVG` | Robust to import order (attribute lookup at call time), but changes global `ReduceOp.AVG` semantics for the whole process. Real design smell, documented in `patches/reduce_avg_allreduce_patch_xpu.py`. This is the one piece here still worth a narrow core-hook ask if the process-wide semantics change is judged unacceptable on review. |

Net: both capabilities work today with zero `verl-core` changes.

## Why the FSDP2 fix isn't a patch

The first draft of this plan (matching the original proposal doc) was to
wrap `verl.utils.fsdp_utils.apply_fsdp2`: call the original, then walk
`model.modules()` and call `set_force_sum_reduction_for_comms(True)`.

That doesn't actually work reliably: `verl/workers/engine/fsdp/
transformer_impl.py` does `from verl.utils.fsdp_utils import apply_fsdp2` at
its own import time. Reassigning `verl.utils.fsdp_utils.apply_fsdp2` after
that has already happened has no effect on the name bound inside
`transformer_impl.py`.

This plugin doesn't need that patch at all, though: `fsdp_xpu.py`'s engine
subclasses already call `set_force_sum_reduction_for_comms` on the
constructed model *after* `super().initialize()` returns — from inside their
own `EngineRegistry`-registered class, which verl-core calls directly (no
patching involved). This branch extends that existing, already-real
mechanism to loop over every FSDP2-wrapped submodule instead of only the
root (`_force_sum_reduction_on_all_fsdp_modules` in `fsdp_xpu.py`), achieved
here with no patch and no hook.

Note: PR #22's current `fsdp_xpu.py` (the hook-based branch) only fixes a
`self.model`→`self.module` attribute bug and still calls
`set_force_sum_reduction_for_comms` on the root only — it does not yet
include this per-layer loop. Whichever branch ships, this fix should land;
it isn't specific to the monkeypatch-vs-hook question.

## Comparison

| | Files touched outside this plugin | Depends on #7917 merging |
|---|---|---|
| PR #22 (`feature/xpu-vtune-avg-attention`) | 0 (but the `is_reduce_avg_supported`/`attention_utils_module` hooks are dead code without #7917) | Yes |
| This branch | 0 | No |

## Not yet done

- No hardware validation of the `attention_utils_module` patch or the raw
  `dist.all_reduce(AVG)` patch — reasoned through against `verl-core`'s real
  `main` source, not run on a B60. The FSDP2 per-layer fix itself is already
  hardware-validated (see the 12-run A/B cited above); only its
  monkeypatch-vs-engine-override framing here is new.
- No unit tests yet for `patches/`.
- Whether to still ask for a narrow core hook covering the 3 raw
  `dist.all_reduce(AVG)` call sites, given the process-wide semantics change
  this patch makes, is an open question for review.
