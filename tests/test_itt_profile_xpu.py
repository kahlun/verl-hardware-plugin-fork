# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for the Intel VTune (ITT) profiling backend.

ITT ranges are a push/pop stack per thread. A range that is pushed but never
popped does not just lose one measurement -- every later range on that thread
nests inside the leaked one, so the trace is unreadable exactly in the runs
where something went wrong and you most want to read it. These tests pin the
push/pop balance when the timed body raises.
"""

from unittest import mock

import pytest

from verl_hardware_plugin.profilers import itt_profile_xpu


class _ITTSpy:
    """Stand-in for torch.profiler.itt that tracks push/pop depth."""

    def __init__(self):
        self.depth = 0
        self.messages = []

    def range_push(self, message):
        self.messages.append(message)
        self.depth += 1
        return self.depth

    def range_pop(self):
        self.depth -= 1
        return self.depth


@pytest.fixture
def itt():
    spy = _ITTSpy()
    with mock.patch.object(itt_profile_xpu, "_itt", spy):
        yield spy


def test_marked_timer_pops_range_on_success(itt):
    timing = {}
    with itt_profile_xpu.marked_timer("step", timing):
        pass
    assert itt.depth == 0
    assert itt.messages == ["step"]
    assert "step" in timing


def test_marked_timer_pops_range_when_body_raises(itt):
    timing = {}
    with pytest.raises(ValueError, match="boom"):
        with itt_profile_xpu.marked_timer("step", timing):
            raise ValueError("boom")
    assert itt.depth == 0, "a raising body must not leave the ITT range open"


def test_marked_timer_propagates_the_original_exception(itt):
    """The timing exception must survive the finally, not be replaced by it."""
    with pytest.raises(KeyError):
        with itt_profile_xpu.marked_timer("step", {}):
            raise KeyError("original")


def test_annotate_pops_range_when_decorated_function_raises(itt):
    profiler = itt_profile_xpu.VtuneProfiler(rank=0, config=None, tool_config=None)

    @profiler.annotate(message="rollout")
    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        boom()
    assert itt.depth == 0, "a raising decorated function must not leave the ITT range open"
    assert itt.messages == ["rollout"]


def test_annotate_returns_value_and_pops_range_on_success(itt):
    profiler = itt_profile_xpu.VtuneProfiler(rank=0, config=None, tool_config=None)

    @profiler.annotate()
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert itt.depth == 0
    assert itt.messages == ["add"]


def test_annotate_stops_discrete_collection_when_function_raises(itt):
    """With discrete=True the platform profiler is started per call; it must be
    stopped on the error path too, or the collector keeps running forever."""
    profiler = itt_profile_xpu.VtuneProfiler(rank=0, config=None, tool_config=None)
    profiler.discrete = True

    platform = mock.MagicMock()
    with mock.patch.object(itt_profile_xpu, "get_platform", return_value=platform):

        @profiler.annotate()
        def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            boom()

    platform.profiler_start.assert_called_once()
    platform.profiler_stop.assert_called_once()
    assert itt.depth == 0
