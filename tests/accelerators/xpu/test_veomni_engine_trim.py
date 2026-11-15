# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU-only test of the VeOmni XPU engine's cache-trim threshold.

Kept apart from test_veomni_upstream_bridge.py, whose stand-ins for VeOmni modules leave torch's lazy
imports half-initialised for a later real import of the engine; this file sorts first.
"""

import types

import pytest
import torch


def test_veomni_cache_trim_threshold_scales_with_card(monkeypatch):
    """The trim threshold is a fraction of the card unless set in GiB."""
    try:
        from verl_hardware_plugin.accelerators.xpu.engines import veomni_xpu
    except Exception as exc:  # verl / VeOmni / transformers not installed in this environment
        pytest.skip(f"verl's VeOmni engine is not importable here: {type(exc).__name__}: {exc}")

    gib = 1024**3
    props = types.SimpleNamespace(total_memory=24 * gib)
    fake_xpu = types.SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda i: props)
    monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
    monkeypatch.delenv("VERL_XPU_EMPTY_CACHE_RESERVED_GB", raising=False)
    monkeypatch.delenv("VERL_XPU_EMPTY_CACHE_RESERVED_FRACTION", raising=False)
    assert veomni_xpu._trim_threshold_bytes() == pytest.approx(8 * gib)  # B60: same as the old fixed 8 GiB
    props.total_memory = 152 * gib
    assert veomni_xpu._trim_threshold_bytes() == pytest.approx(152 * gib / 3)  # CRI
    monkeypatch.setenv("VERL_XPU_EMPTY_CACHE_RESERVED_FRACTION", "0.5")
    assert veomni_xpu._trim_threshold_bytes() == pytest.approx(76 * gib)
    monkeypatch.setenv("VERL_XPU_EMPTY_CACHE_RESERVED_GB", "8")
    assert veomni_xpu._trim_threshold_bytes() == pytest.approx(8 * gib)
