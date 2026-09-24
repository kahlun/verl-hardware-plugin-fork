# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Intel VTune (ITT) profiling backend for verl.

Wired in via verl_hardware_plugin.patches.dist_profiler_patch_xpu, which
monkeypatches DistProfiler.__init__ to select VtuneProfiler when
`profiler.tool: vtune` -- not via a PlatformBase hook. See that module's
docstring for what this patch does and does not cover (ambient
marked_timer()/mark_start_range() calls elsewhere in verl-core's trainer
code are NOT covered; only VtuneProfiler's own start/stop/annotate are).
"""

import functools
from contextlib import contextmanager
from typing import Callable, Optional

import torch.profiler.itt as _itt

from verl.plugin.platform import get_platform
from verl.utils.profiler.config import NsightToolConfig
from verl.utils.profiler.profile import DistProfiler, ProfilerConfig


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> int:
    """Push an ITT range onto the stack. Returns the stack depth (used as range_id).

    color/domain/category are accepted for API compatibility but ignored — ITT does not
    support them.
    """
    return _itt.range_push(message or "")


def mark_end_range(range_id: int) -> None:
    """Pop the innermost ITT range. range_id is accepted for API compatibility."""
    _itt.range_pop()


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorate a function to wrap its execution in an ITT range."""

    def decorator(func):
        profile_message = message or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _itt.range_push(profile_message)
            try:
                return func(*args, **kwargs)
            finally:
                _itt.range_pop()

        return wrapper

    return decorator


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with ITT ranges (Intel VTune).

    Measures execution time, accumulates into timing_raw, and emits an ITT range
    visible in VTune when the process runs under a collector.

    The range is popped in a ``finally`` so a raising body cannot leave it open.
    (verl-core's nvtx equivalent does not do this; an exception there leaks an
    NVTX range and every later range on the thread nests inside it, which makes
    the resulting trace unreadable exactly when you most want to read it.)
    """
    mark_range = mark_start_range(message=name)
    from verl.utils.profiler.performance import _timer

    try:
        yield from _timer(name, timing_raw)
    finally:
        mark_end_range(mark_range)


class VtuneProfiler(DistProfiler):
    """Intel VTune profiler. Installed in a worker to control ITT ranges.

    Unlike Nsight (which has a process-level start/stop API), VTune attaches
    externally as a collector and observes range_push/range_pop events, so
    PlatformXPU.profiler_start/profiler_stop are no-ops by design — the ranges
    emitted by mark_start_range/mark_end_range are the real signal.

    Because those two hooks are no-ops, ``tool_config.discrete`` has no effect on
    XPU: both values behave identically. It is read only to keep the DistProfiler
    contract, so a missing tool_config is not an error here.
    """

    def __init__(self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[NsightToolConfig], **kwargs):
        if not config:
            config = ProfilerConfig(ranks=[])
        # verl core has no `tool_config.vtune` schema entry, so `tool_config.get("vtune")`
        # resolves to None and DistProfiler then substitutes the whole tool_config mapping
        # (verl/utils/profiler/profile.py: `if tool_config is None: tool_config = config.tool_config`).
        # That mapping is truthy but has no `discrete`, so read it defensively — the same way
        # core itself does. `discrete` is a no-op on XPU anyway, so False is always correct.
        self.discrete: bool = getattr(tool_config, "discrete", False)

    def start(self, **kwargs):
        if not self.discrete:
            get_platform().profiler_start()

    def stop(self):
        if not self.discrete:
            get_platform().profiler_stop()

    def step(self):
        return

    def annotate(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                profile_name = message or func.__name__

                if self.discrete:
                    get_platform().profiler_start()
                mark_range = mark_start_range(message=profile_name)

                try:
                    return func(*args, **kwargs_inner)
                finally:
                    # Pop the range even if func raised, or every later range on
                    # this thread nests inside the leaked one.
                    mark_end_range(mark_range)
                    if self.discrete:
                        get_platform().profiler_stop()

            return wrapper

        return decorator
