# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Plugin-side monkeypatches for Intel XPU, applied with zero changes to
verl-core and with no dependency on any new PlatformBase hook.

This package exists as the "how far can pure monkeypatch actually go"
counterpart to the PlatformBase-hook approach used elsewhere in this plugin
(is_reduce_avg_supported/attention_utils_module/profiler_markers/
dist_profiler_cls, gated behind the still-unmerged verl-project/verl#7917).
See docs/design/xpu-monkeypatch-experiment.md for the per-capability verdict
and why two capabilities are NOT here (they are genuine blockers, not solved
by patching harder).

Each patch module is independently guarded by an XPU-availability check, and
apply_all() is idempotent, so importing this package is safe even when
another vendor's plugin (or no accelerator at all) is active in the same
process.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_all() -> None:
    """Apply every XPU monkeypatch. Safe to call multiple times or on non-XPU hosts."""
    from . import attention_patch_xpu, dist_profiler_patch_xpu, reduce_avg_allreduce_patch_xpu

    for module in (attention_patch_xpu, dist_profiler_patch_xpu, reduce_avg_allreduce_patch_xpu):
        try:
            module.apply()
        except Exception as e:
            logger.debug("%s.apply() skipped: %s", module.__name__, e)
