# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch torch.distributed.all_reduce(op=AVG) -> SUM + manual divide,
process-wide, for Intel XPU (xccl does not implement ReduceOp.AVG).

This is the one patch in this package worth calling "doable, but with a real
design smell" rather than "clean":

- verl/workers/engine_workers.py, verl/trainer/sft_trainer.py, and
  verl/utils/profiler/performance.py::reduce_timing each call
  `torch.distributed.all_reduce(x, op=ReduceOp.AVG, ...)` as a literal inline
  expression -- there is no per-site wrapper or dispatch object to patch, the
  way there is for attention padding.
- Patching `torch.distributed.all_reduce` itself is robust to import order
  (every call site does `dist.all_reduce(...)`, an attribute lookup on the
  `torch.distributed` module performed at call time, not at each file's own
  import time), so this does take effect regardless of load order.
- The cost is real: this changes the process-wide meaning of
  `ReduceOp.AVG` for every call site in the process, XPU-related or not.
  Anyone reading verl-core's source sees real `ReduceOp.AVG` semantics and
  silently gets SUM+divide instead. That is a legitimate argument for fixing
  this in verl-core instead (verl-project/verl#7917's
  `is_reduce_avg_supported()` hook covers exactly these 3 call sites) -- this
  patch exists to show it is *possible* without core changes, not to claim
  it is the better design.

async_op=True is intentionally unsupported: correctly dividing the result
requires the collective to have already completed, which async_op explicitly
defers. Call sites using async_op=True with op=AVG fall through to the
original (broken-on-xccl) behavior rather than silently producing a wrong
answer with no error.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False


def _xpu_available() -> bool:
    import torch

    return hasattr(torch, "xpu") and torch.xpu.is_available()


def apply() -> None:
    global _applied
    if _applied:
        return
    if not _xpu_available():
        return

    import torch.distributed as dist

    original_all_reduce = dist.all_reduce

    def _patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op != dist.ReduceOp.AVG or async_op:
            return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

        world_size = dist.get_world_size(group=group)
        result = original_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
        tensor.div_(world_size)
        return result

    dist.all_reduce = _patched_all_reduce
    _applied = True
    logger.info("[verl_hardware_plugin] Patched torch.distributed.all_reduce(op=AVG) for XPU/xccl")
