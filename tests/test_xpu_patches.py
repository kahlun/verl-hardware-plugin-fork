# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for verl_hardware_plugin.patches (Intel XPU monkeypatches).

Each patch in that package mutates global verl-core/torch state, so the two
things worth pinning down are:

1. On a non-XPU host every patch is a no-op, and nothing global is touched.
   (A CPU/CUDA/NPU process still imports this plugin -- the entry point is
   discovered on every host.)
2. On an XPU host a patch that fails to install is not swallowed: a
   correctness-critical one raises, a feature one is logged loudly.

These run on a CPU-only host; XPU presence is simulated by patching the shared
guard, so no Intel GPU is required.
"""

import logging
from contextlib import ExitStack
from unittest import mock

import pytest

from verl_hardware_plugin import patches
from verl_hardware_plugin.patches import (
    attention_patch_xpu,
    dist_profiler_patch_xpu,
    reduce_avg_allreduce_patch_xpu,
)

PATCH_MODULES = (attention_patch_xpu, dist_profiler_patch_xpu, reduce_avg_allreduce_patch_xpu)


@pytest.fixture(autouse=True)
def _reset_applied_flags():
    """Patches are idempotent via a module-level _applied flag; reset it per test."""
    originals = {m: m._applied for m in PATCH_MODULES}
    for m in PATCH_MODULES:
        m._applied = False
    yield
    for m, value in originals.items():
        m._applied = value


def _force_xpu_availability(available: bool):
    """Patch the shared guard as each module sees it (all import it by name)."""
    stack = ExitStack()
    for target in (patches, *PATCH_MODULES):
        stack.enter_context(mock.patch.object(target, "xpu_available", return_value=available))
    return stack


@pytest.fixture
def no_xpu():
    with _force_xpu_availability(False):
        yield


@pytest.fixture
def fake_xpu():
    with _force_xpu_availability(True):
        yield


@pytest.mark.parametrize("module", PATCH_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_patch_is_noop_without_xpu(module, no_xpu):
    """Every patch module must check for XPU before mutating anything global.

    dist_profiler_patch_xpu originally did not, so importing this plugin on a
    CUDA host rerouted that host's `profiler.tool: vtune` to the Intel ITT
    profiler.
    """
    module.apply()
    assert module._applied is False, f"{module.__name__} installed itself on a non-XPU host"


def test_dist_profiler_untouched_without_xpu(no_xpu):
    """The concrete consequence of the guard above, checked on the real class."""
    from verl.utils.profiler.profile import DistProfiler

    before = DistProfiler.__init__
    dist_profiler_patch_xpu.apply()
    assert DistProfiler.__init__ is before


def test_all_reduce_untouched_without_xpu(no_xpu):
    import torch.distributed as dist

    before = dist.all_reduce
    reduce_avg_allreduce_patch_xpu.apply()
    assert dist.all_reduce is before


def test_apply_all_is_quiet_on_non_xpu_host(no_xpu, caplog):
    """A failing patch on a non-XPU host is uninteresting: debug-log, no raise."""
    with mock.patch.object(attention_patch_xpu, "apply", side_effect=ImportError("boom")):
        with caplog.at_level(logging.DEBUG, logger="verl_hardware_plugin.patches"):
            patches.apply_all()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_apply_all_raises_when_required_patch_fails_on_xpu(fake_xpu):
    """A correctness-critical patch that did not install must not be swallowed.

    Without this, an XPU run continues with real ReduceOp.AVG on xccl and
    silently produces wrong gradients.
    """
    with mock.patch.object(reduce_avg_allreduce_patch_xpu, "apply", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="reduce_avg_allreduce_patch_xpu"):
            patches.apply_all()


def test_apply_all_reports_but_survives_optional_patch_failure_on_xpu(fake_xpu, caplog):
    """The vtune patch is observability, not correctness: log loudly, keep going."""
    with mock.patch.object(dist_profiler_patch_xpu, "apply", side_effect=RuntimeError("boom")):
        with mock.patch.object(attention_patch_xpu, "apply"):
            with mock.patch.object(reduce_avg_allreduce_patch_xpu, "apply"):
                with caplog.at_level(logging.DEBUG, logger="verl_hardware_plugin.patches"):
                    patches.apply_all()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an optional patch failing on an XPU host must be visible above debug level"
    assert "dist_profiler_patch_xpu" in errors[0].getMessage()
