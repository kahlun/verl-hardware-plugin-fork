# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Plugin-side monkeypatch for Intel XPU reduce_avg support, applied with
zero changes to verl-core and with no dependency on any new PlatformBase
hook.

Per the maintainer's ruling on verl-hardware-plugin#26 (2026-09-23): default
new hardware backends to plugin-side monkeypatching, reserving new
PlatformBase hooks for capabilities that genuinely can't be done from the
plugin. `attention_utils_module()` passed that bar and stays a hook
(implemented in `platforms/platform_xpu.py`, see PR #22) — this package only
covers `is_reduce_avg_supported`, the one capability that didn't. See
docs/design/xpu-monkeypatch-reduce-avg.md for the full reasoning.

The VTune/dist_profiler_cls monkeypatch is intentionally not part of this
package — see docs/design/xpu-monkeypatch-vtune-profiler.md, developed as a
separate branch/PR so the two can be reviewed and merged independently.

The patch module is guarded by an XPU-availability check, and apply_all() is
idempotent, so importing this package is safe even when another vendor's
plugin (or no accelerator at all) is active in the same process.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_all() -> None:
    """Apply every XPU monkeypatch. Safe to call multiple times or on non-XPU hosts."""
    from . import reduce_avg_allreduce_patch_xpu

    for module in (reduce_avg_allreduce_patch_xpu,):
        try:
            module.apply()
        except Exception as e:
            logger.debug("%s.apply() skipped: %s", module.__name__, e)
