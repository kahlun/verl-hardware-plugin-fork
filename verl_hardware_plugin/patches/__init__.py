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

Every patch module is independently guarded by the shared XPU-availability
check in ``_xpu_guard`` and sets its own ``_applied`` flag, so apply_all() is
idempotent and importing this package is safe even when another vendor's
plugin (or no accelerator at all) is active in the same process.

Failure handling
    On a non-XPU process there is nothing to install, so a failure is not
    interesting and is logged at debug. On an XPU process a patch that fails
    to install means an advertised capability is silently absent, which is
    exactly the kind of thing that later surfaces as a wrong number or an
    unexplained collective hang -- so those are never swallowed:

    - REQUIRED patches guard numerical correctness (oneCCL's ReduceOp.AVG is
      not reliably available -- see reduce_avg_allreduce_patch_xpu's docstring
      for why "not reliably" rather than "not at all"; rmpad attention needs
      the XPU function set). If one of these cannot be installed, training
      would produce wrong results or abort far from the cause, so apply_all()
      re-raises as RuntimeError.
    - Non-required patches are feature wiring (`profiler.tool: vtune`
      selection). Losing one degrades observability but not correctness, so it
      is reported with logger.exception and training continues.
"""

import importlib
import logging
import os

from ._xpu_guard import xpu_available

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# (module attribute name, required-for-correctness)
_PATCH_MODULES = (
    ("reduce_avg_allreduce_patch_xpu", True),
    ("attention_patch_xpu", True),
    ("dist_profiler_patch_xpu", False),
)


def apply_all() -> None:
    """Apply every XPU monkeypatch. Safe to call multiple times or on non-XPU hosts.

    Raises:
        RuntimeError: on an XPU host, if a correctness-critical patch could not
            be installed. Never raises on a non-XPU host.
    """
    on_xpu = xpu_available()

    for name, required in _PATCH_MODULES:
        qualname = f"{__name__}.{name}"
        try:
            # Imported here, not at module scope, so that a module-level import
            # failure is subject to the same policy as an apply() failure.
            importlib.import_module(f".{name}", __name__).apply()
        except Exception as e:
            if not on_xpu:
                # Nothing was going to be installed on this host anyway.
                logger.debug("%s.apply() skipped on non-XPU host: %s", qualname, e)
            elif required:
                raise RuntimeError(
                    f"{qualname}.apply() failed on an Intel XPU host. This patch is required for correct "
                    f"results (oneCCL's ReduceOp.AVG is not reliably available / rmpad attention needs the "
                    f"XPU function set); continuing would silently produce wrong numbers."
                ) from e
            else:
                logger.exception(
                    "%s.apply() failed on an Intel XPU host; the capability it provides is unavailable",
                    qualname,
                )
