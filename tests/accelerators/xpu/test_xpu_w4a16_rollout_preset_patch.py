# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for the W4A16 rollout-quantization preset patch. Mock only -- no accelerator."""

import sys
import types

import pytest

from verl_hardware_plugin.accelerators.xpu.patches import w4a16_rollout_preset_patch as patch


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(patch, "_xpu_available", lambda: True)
    patch._reset()
    yield
    patch._reset()


def _fake_module(monkeypatch, name="fake_compressed_tensors"):
    module = types.ModuleType(name)
    module.SUPPORTED_SCHEMES = {"FP8_BLOCK": "fp8", "NVFP4": "nvfp4"}
    monkeypatch.setattr(patch, "_TARGET", name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_adds_preset_without_qat_mode_and_keeps_existing(monkeypatch):
    module = _fake_module(monkeypatch)
    patch.apply()
    assert module.SUPPORTED_SCHEMES == {"FP8_BLOCK": "fp8", "NVFP4": "nvfp4", "W4A16": None}


def test_idempotent(monkeypatch):
    module = _fake_module(monkeypatch)
    patch.apply()
    module.SUPPORTED_SCHEMES["W4A16"] = "custom"
    patch.apply()
    assert module.SUPPORTED_SCHEMES["W4A16"] == "custom"


def test_does_nothing_without_xpu(monkeypatch):
    module = _fake_module(monkeypatch)
    monkeypatch.setattr(patch, "_xpu_available", lambda: False)
    patch.apply()
    assert "W4A16" not in module.SUPPORTED_SCHEMES
    assert not any(isinstance(f, patch._AfterImport) for f in sys.meta_path)


def test_registers_right_after_a_later_import(monkeypatch, tmp_path):
    name = "w4a16_patch_late_target"
    (tmp_path / f"{name}.py").write_text("SUPPORTED_SCHEMES = {'FP8_BLOCK': 'fp8'}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(patch, "_TARGET", name)
    monkeypatch.delitem(sys.modules, name, raising=False)

    patch.apply()
    assert name not in sys.modules  # not imported early
    assert any(isinstance(f, patch._AfterImport) for f in sys.meta_path)

    import importlib

    module = importlib.import_module(name)
    try:
        assert module.SUPPORTED_SCHEMES == {"FP8_BLOCK": "fp8", "W4A16": None}
        assert not any(isinstance(f, patch._AfterImport) for f in sys.meta_path)  # one-shot
    finally:
        sys.modules.pop(name, None)


def test_real_verl_accepts_the_preset():
    pytest.importorskip("compressed_tensors")
    compressed_tensors = pytest.importorskip("verl.utils.qat.compressed_tensors")
    before = dict(compressed_tensors.SUPPORTED_SCHEMES)
    try:
        patch.apply()
        config = compressed_tensors.build_compressed_tensors_config(linear="W4A16", moe=None, ignore=[], config_file=None)
        group = next(iter(config["config_groups"].values()))
        assert group["format"] == "pack-quantized"
        assert group["weights"]["num_bits"] == 4 and group["weights"]["group_size"] == 128
    finally:
        compressed_tensors.SUPPORTED_SCHEMES.clear()
        compressed_tensors.SUPPORTED_SCHEMES.update(before)
