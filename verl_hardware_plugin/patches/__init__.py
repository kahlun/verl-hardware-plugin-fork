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
covers `is_reduce_avg_supported`, the one capability that didn't.

Call `apply_all()` from `PlatformXPU.__init__`, not from this plugin's
top-level `verl_hardware_plugin/__init__.py`: the latter runs on any host
where XPU merely happens to be importable, regardless of which platform
verl actually selects for the run, whereas `PlatformXPU()` is only
constructed once XPU is the selected platform (see
`verl.plugin.platform.platform_manager._create_platform`).

The VTune/dist_profiler_cls monkeypatch is a separate, unrelated concern and
intentionally not part of this package — it ships as its own branch/PR so
the two can be reviewed and merged independently.

The patch module is guarded by an XPU-availability check, and apply_all() is
idempotent, so calling it is safe even when another vendor's platform (or no
accelerator at all) ends up selected in the same process.
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
