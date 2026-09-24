# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch DistProfiler.__init__ so `profiler.tool: vtune` resolves to
VtuneProfiler, with zero verl-core changes and no dependency on
PlatformBase.dist_profiler_cls().

Why patching the class method (not reassigning the name `DistProfiler`)
matters:
    verl/utils/profiler/profile.py::DistProfiler.__init__ has a hardcoded
    if/elif chain on `self._tool` ("nsys", "npu", "torch", "torch_memory",
    "precision_debugger") with no registry -- there is no hook to add a
    "vtune" branch to. But `DistProfiler` is a class object shared by every
    caller regardless of how or when they imported it (`from
    verl.utils.profiler import DistProfiler` vs `verl.utils.profiler.profile.
    DistProfiler`) -- reassigning its `__init__` attribute mutates that one
    shared object, so it takes effect for every existing and future caller.
    That is different from e.g. reduce_avg's `apply_fsdp2`, which is a plain
    function imported by name into another module's namespace before this
    plugin gets a chance to patch it.

Wrap-then-fixup is safe here because DistProfiler.__init__ does not have side
effects beyond constructing `self._impl`: run the real __init__, then swap
`self._impl` for VtuneProfiler if the real __init__ fell back to
`_NoOpProfiler` for a tool name it doesn't recognize and that name is
"vtune". Nothing destructive happens in the original that a post-hoc fixup
could not undo.

This does NOT solve verl-project/verl-hardware-plugin#26's
`profiler_markers` hook: the ambient `marked_timer()`/`mark_start_range()`
calls scattered through verl-core's own trainer code (verl/utils/profiler/
__init__.py's import-time module selection) are a separate, genuine blocker
-- see docs/design/xpu-monkeypatch-experiment.md. What this patch gets you:
`profiler.tool: vtune` correctly selects VtuneProfiler, and its own
start()/stop()/annotate() emit real ITT ranges (VtuneProfiler calls its own
local mark_start_range/mark_end_range, not the ambient ones). What it does
NOT get you: the rest of the trainer's own `marked_timer()` calls elsewhere
in verl-core emitting ITT ranges too.
"""

import logging
import os

from ._xpu_guard import xpu_available

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    # DistProfiler is a process-wide shared class object, so patching it on a
    # CPU/CUDA/NPU process would reroute *that* platform's `tool: vtune` to the
    # Intel ITT implementation. Same guard as every other patch in this package.
    if not xpu_available():
        return

    from verl.utils.profiler.profile import DistProfiler, _NoOpProfiler

    original_init = DistProfiler.__init__

    def _patched_init(self, rank, config=None, tool_config=None, save_file_prefix=None, **kwargs):
        original_init(self, rank, config=config, tool_config=tool_config, save_file_prefix=save_file_prefix, **kwargs)

        if self._tool == "vtune" and isinstance(self._impl, _NoOpProfiler):
            from verl_hardware_plugin.profilers.itt_profile_xpu import VtuneProfiler

            self._impl = VtuneProfiler(rank=rank, config=self.config, tool_config=self.tool_config, **kwargs)

    DistProfiler.__init__ = _patched_init
    _applied = True
    logger.info("[verl_hardware_plugin] Patched DistProfiler.__init__: +vtune")
