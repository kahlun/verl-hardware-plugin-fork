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

## Comparison

| | Files touched outside this plugin | Depends on #7917 merging |
|---|---|---|
| PR #22 (`feature/xpu-vtune-avg-attention`) | 0 (but 4 `PlatformXPU` hook methods are dead code without #7917) | Yes, for all 4 hooks |
| This branch | 0 | No, for 3 of 4. `profiler_markers`'s ambient markers and the rollout-server allowlist remain open — either fork ~40 lines of server code into the plugin, or ask for the two small core changes verl-hardware-plugin#26 already discusses. |

## Failure policy for the patches

Every module in `patches/` shares one XPU guard (`patches/_xpu_guard.py`) and
is a hard no-op on a CPU/CUDA/NPU process — the entry point is discovered on
every host, not just Intel ones, and all three patches mutate process-global
state (`torch.distributed.all_reduce`, `verl.utils.attention_utils.
_get_attention_functions`, `DistProfiler.__init__`). Patching any of those on
another vendor's host changes *that* vendor's behavior.

`apply_all()` splits failures by consequence rather than swallowing them:

| Patch | On failure, on an XPU host |
|---|---|
| `reduce_avg_allreduce_patch_xpu` | **raise** — xccl cannot execute `ReduceOp.AVG`; continuing produces wrong gradients with no error |
| `attention_patch_xpu` | **raise** — rmpad attention needs the XPU function set |
| `dist_profiler_patch_xpu` | `logger.exception`, continue — losing `tool: vtune` costs observability, not correctness |

On a non-XPU host nothing was going to install anyway, so a failure is
debug-logged and never raised.

## Not yet done

- Hardware validation status: see the PR's test-plan checklist. The "clean"
  verdicts above were reasoned against `verl-core`'s real `main` source; read
  them as "verified against source" until a B60 run is recorded there.
- No unit tests yet for the two capabilities this branch deliberately does
  *not* patch (they are argued from source, not exercised).
