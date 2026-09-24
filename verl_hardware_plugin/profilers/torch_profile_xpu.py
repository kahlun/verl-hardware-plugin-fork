# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkey-patch verl's torch profiler to support Intel XPU.

Patched: ``TorchProfilerToolConfig`` accepts ``xpu`` and ``get_torch_profiler``
collects ``torch.profiler.ProfilerActivity.XPU`` when requested. Unlike
``torch_profile_mlu.py`` (which fully replaces both), these patches wrap
whatever is currently installed -- composing with the MLU patch (or any
future vendor patch) instead of clobbering it, and tracking verl-core's
current ``get_torch_profiler`` signature/behavior automatically instead of
freezing a copy of it that goes stale as verl-core evolves.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

_original_get_torch_profiler = None
_original_post_init = None


def _patch_tool_config():
    """Add 'xpu' to allowed contents in TorchProfilerToolConfig, without
    disturbing any other content name a different patch already allows."""
    from verl.utils.profiler.config import TorchProfilerToolConfig

    global _original_post_init
    if _original_post_init is not None:
        return

    _original_post_init = TorchProfilerToolConfig.__post_init__

    @functools.wraps(_original_post_init)
    def _patched_post_init(self):
        contents = self.contents
        if isinstance(contents, list) and "xpu" in contents:
            # Hide "xpu" from the wrapped validator (whatever it currently is), then
            # restore the full list so get_torch_profiler still sees "xpu" in contents.
            # Bypass BaseConfig.__setattr__'s frozen-field check: this dataclass field
            # already exists in __dict__, so a normal assignment would raise
            # FrozenInstanceError even for this same-value round trip.
            object.__setattr__(self, "contents", [c for c in contents if c != "xpu"])
            try:
                _original_post_init(self)
            finally:
                object.__setattr__(self, "contents", contents)
        else:
            _original_post_init(self)

    TorchProfilerToolConfig.__post_init__ = _patched_post_init
    logger.info("[verl_hardware_plugin] Patched TorchProfilerToolConfig: +xpu")


def _patch_get_torch_profiler():
    """Wrap the currently-installed get_torch_profiler to also collect XPU activity."""
    import verl.utils.profiler.torch_profile as tp

    global _original_get_torch_profiler
    if _original_get_torch_profiler is not None:
        return

    _original_get_torch_profiler = tp.get_torch_profiler

    @functools.wraps(_original_get_torch_profiler)
    def _xpu_get_torch_profiler(contents, *args, **kwargs):
        prof = _original_get_torch_profiler(contents, *args, **kwargs)
        if hasattr(torch.profiler.ProfilerActivity, "XPU") and (not contents or "xpu" in contents):
            activities = list(getattr(prof, "activities", None) or [])
            activities.append(torch.profiler.ProfilerActivity.XPU)
            prof.activities = activities
        return prof

    tp.get_torch_profiler = _xpu_get_torch_profiler
    logger.info("[verl_hardware_plugin] Patched get_torch_profiler: +XPU activity")
