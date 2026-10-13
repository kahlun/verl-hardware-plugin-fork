# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Regression test for the Qwen2-VL/GLM-4V flash-attn HF-Hub-kernels fallback patch.

verl/models/transformers/qwen2_vl.py and glm4v.py only bind `flash_attn_func`
when `is_flash_attn_2_available()` or `is_npu_available` fires. Neither does on
XPU, so without this patch `_custom_flash_attention_forward` references an
unbound name the moment `use_remove_padding=True` or `ulysses_sp_size>1`
triggers the monkeypatched attention forward. These tests exercise the patch
against the real installed verl-core modules (not fakes), so a change to
either file's structure that this patch depends on shows up here.
"""

import pytest

import verl.models.transformers.glm4v as glm4v
import verl.models.transformers.qwen2_vl as qwen2_vl
from verl_hardware_plugin.accelerators.xpu.patches import vl_attn_hub_kernels_patch as patch_mod


@pytest.fixture(params=[qwen2_vl, glm4v])
def target_module(request, monkeypatch):
    """A model module with flash_attn_func unbound, as it is on real XPU."""
    module = request.param
    monkeypatch.delattr(module, "flash_attn_func", raising=False)
    monkeypatch.delattr(module, "flash_attn_varlen_func", raising=False)
    original_forward = module._custom_flash_attention_forward
    yield module
    module._custom_flash_attention_forward = original_forward


def test_skips_module_that_already_has_a_backend_bound(target_module, monkeypatch):
    """If CUDA/NPU already bound flash_attn_func, the patch must not override it."""
    sentinel = object()
    monkeypatch.setattr(target_module, "flash_attn_func", sentinel, raising=False)

    patch_mod._patch_module(target_module)

    assert target_module.flash_attn_func is sentinel


def test_binds_hub_kernels_fallback_when_available(target_module, monkeypatch):
    def fake_fallback():
        def flash_attn_func(*args, window_size=None, deterministic=None, **kwargs):
            return "fake-output"

        def flash_attn_varlen_func(*args, **kwargs):
            return "fake-varlen-output"

        return flash_attn_func, flash_attn_varlen_func

    monkeypatch.setattr(patch_mod, "_load_hub_kernels_fallback", fake_fallback)

    patch_mod._patch_module(target_module)

    assert target_module.flash_attn_func is not None
    assert target_module._flash_supports_window_size is True
    assert target_module._flash_supports_deterministic is True
    # Original forward is left untouched - the fallback kernel is enough.
    assert target_module._custom_flash_attention_forward.__name__ == "_custom_flash_attention_forward"


def test_raises_clear_error_when_no_backend_is_available_anywhere(target_module, monkeypatch):
    monkeypatch.setattr(patch_mod, "_load_hub_kernels_fallback", lambda: None)

    patch_mod._patch_module(target_module)

    with pytest.raises(RuntimeError, match="no flash-attn backend is available"):
        target_module._custom_flash_attention_forward()


def test_apply_is_a_noop_off_xpu(monkeypatch):
    monkeypatch.setattr(patch_mod, "_xpu_available", lambda: False)
    monkeypatch.setattr(patch_mod, "_applied", False)
    called = []
    monkeypatch.setattr(patch_mod, "_patch_module", lambda module: called.append(module))

    patch_mod.apply()

    assert called == []
