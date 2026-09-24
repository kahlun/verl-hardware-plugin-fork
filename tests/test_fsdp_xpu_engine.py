# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for the Intel XPU FSDP engine's xccl reduce workaround.

Two things are pinned here:

1. The attribute name. verl's FSDPEngine assigns the wrapped model to
   ``self.module`` (transformer_impl.py::_build_model_optimizer, called from
   ``initialize()``); there is no ``self.model`` on these engines, only
   ``self.model_config``.
   The engine previously guarded on ``hasattr(self.model, ...)``, so the
   workaround silently never ran and xccl kept getting ReduceOp.AVG.
2. The recursion. fully_shard() wraps each transformer layer as well as the
   root, and each wrapper has its own reduction op, so fixing only the root
   leaves per-layer gradient sync on AVG.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


@pytest.fixture(scope="module")
def force_sum_helper():
    fsdp_xpu = pytest.importorskip("verl_hardware_plugin.engines.fsdp_xpu")
    return fsdp_xpu._force_sum_reduction_on_all_fsdp_modules


class _FakeFSDPModule(nn.Module):
    """Stands in for a fully_shard()-wrapped module: has the setter, records calls."""

    def __init__(self):
        super().__init__()
        self.forced = None

    def set_force_sum_reduction_for_comms(self, value):
        self.forced = value


class _FakeFSDPRoot(_FakeFSDPModule):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeFSDPModule() for _ in range(n_layers)])
        # A plain submodule with no setter must simply be skipped, not crash.
        self.embed = nn.Linear(4, 4)


def test_force_sum_reduction_covers_root_and_every_wrapped_layer(force_sum_helper):
    root = _FakeFSDPRoot(n_layers=3)

    force_sum_helper(root)

    assert root.forced is True
    assert [layer.forced for layer in root.layers] == [True, True, True]


def test_force_sum_reduction_tolerates_a_model_without_the_setter(force_sum_helper):
    """FSDP1 / unwrapped models have no setter at all; this must be a no-op."""
    force_sum_helper(nn.Linear(4, 4))


def test_verl_fsdp_engine_exposes_module_not_model():
    """Guards the attribute contract the workaround depends on.

    If a future verl release renames ``module``, this fails here instead of
    silently disabling the workaround at runtime.
    """
    import inspect

    transformer_impl = pytest.importorskip("verl.workers.engine.fsdp.transformer_impl")
    # Scan the whole class, not initialize(): the assignment lives in
    # _build_model_optimizer(), which initialize() calls.
    class_src = inspect.getsource(transformer_impl.FSDPEngine)
    assert "self.module = " in class_src, "verl's FSDPEngine no longer assigns self.module"
    assert "self.model = " not in class_src, "verl's FSDPEngine now also assigns self.model; revisit the engine"
