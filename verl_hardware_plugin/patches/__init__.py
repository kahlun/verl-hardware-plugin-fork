# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Plugin-side monkeypatches for Intel XPU reduce_avg + attention support,
applied with zero changes to verl-core and with no dependency on any new
PlatformBase hook.

This is the "how far can pure monkeypatch actually go" counterpart to the
PlatformBase-hook approach used elsewhere in this plugin
(is_reduce_avg_supported/attention_utils_module, gated behind the
still-unmerged verl-project/verl#7917). See
docs/design/xpu-monkeypatch-reduce-avg-attention.md for the per-capability
verdict.

The VTune/dist_profiler_cls monkeypatch is intentionally not part of this
package — see docs/design/xpu-monkeypatch-vtune-profiler.md, developed as a
separate branch/PR so the two can be reviewed and merged independently.

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
    from . import attention_patch_xpu, reduce_avg_allreduce_patch_xpu

    for module in (attention_patch_xpu, reduce_avg_allreduce_patch_xpu):
        try:
            module.apply()
        except Exception as e:
            logger.debug("%s.apply() skipped: %s", module.__name__, e)
