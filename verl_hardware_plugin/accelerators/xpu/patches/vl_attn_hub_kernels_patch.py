# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch Qwen2-VL / GLM-4V's flash-attn dispatch to fall back to HF Hub
`kernels` on Intel XPU.

Why this is needed: verl/models/transformers/qwen2_vl.py and glm4v.py only
bind `flash_attn_func`/`flash_attn_varlen_func` (plus the three capability
flags derived from them) inside `if is_flash_attn_2_available():` or
`if is_npu_available:` module-level branches. Neither fires on XPU - the
CUDA-only `flash_attn` wheel doesn't install there and there's no NPU build -
so those names are simply never bound. The monkeypatch that routes attention
through `_custom_flash_attention_forward` is gated on `use_remove_padding` or
`ulysses_sp_size > 1`, not on attention backend, so it fires regardless of
device and the unbound names cause a bare NameError on first forward pass.

FA2 does work on XPU, just through a dispatch path these two files never use:
HF Hub `kernels`, the same mechanism transformers' own
`attn_implementation="flash_attention_2"` resolution already uses in
production for this plugin's image (see
docker/intel_gpu/requirements-intel-gpu.txt). This patch binds the same
fallback the two files are missing, using transformers' own
`get_attn_kernel_version()` so it tracks any future bump to transformers'
hub-kernel version mapping automatically, rather than hardcoding one here.

Scope and alternative considered: this could instead be a verl-core change to
qwen2_vl.py/glm4v.py directly (see kahlun/verl#33) - the gap isn't XPU-named,
it's "neither CUDA nor NPU flash-attn is available", so it reads as
hardware-agnostic. But verl-core has no platform hook for *which flash-attn
backend a model file should bind* (unlike the bert_padding helpers in
attention_utils.py, which already dispatch through
`PlatformBase.get_attention_functions()`), so landing it in core would still
mean two near-identical `if not available: try hub kernels` blocks hardcoded
into model files that otherwise contain no device branching. Keeping it here
as a monkeypatch applied only when XPU is the selected platform avoids
widening verl-core's surface for a gap that today only reproduces on this
one device, consistent with this plugin's default of plugin-side patches
(core PRs reserved for genuine hardware-agnostic blockers).
"""

import importlib
import inspect
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False

# model.config.model_type values whose verl monkey_patch.py dispatch resolves
# to each module (verl/models/transformers/monkey_patch.py); both VL variants
# below reuse their listed module's helpers, there is no separate qwen2_5_vl.py.
_TARGET_MODULES = (
    "verl.models.transformers.qwen2_vl",  # model_type in {"qwen2_vl", "qwen2_5_vl"}
    "verl.models.transformers.glm4v",  # model_type == "glm4v"
)


def _xpu_available() -> bool:
    import torch

    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _load_hub_kernels_fallback():
    """Return (flash_attn_func, flash_attn_varlen_func) via HF Hub kernels, or None."""
    try:
        from kernels import get_kernel
        from transformers.integrations.hub_kernels import get_attn_kernel_version
        from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK

        repo_id = FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"]
        kernel = get_kernel(repo_id, version=get_attn_kernel_version(repo_id))
        return kernel.flash_attn_func, kernel.flash_attn_varlen_func
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[verl_hardware_plugin] Could not load a flash-attn backend via HF Hub kernels: {e}")
        return None


def _patch_module(module) -> None:
    if getattr(module, "flash_attn_func", None) is not None:
        return  # CUDA or NPU branch already bound it - nothing to do

    fallback = _load_hub_kernels_fallback()

    if fallback is None:
        # Still nothing available - fail clearly at the call site instead of
        # a bare NameError deep inside tensor ops.
        def _unavailable_forward(*args, **kwargs):
            raise RuntimeError(
                f"{module.__name__} attention was monkeypatched to the flash-attn forward "
                "(triggered by use_remove_padding=True or ulysses_sp_size>1) but no flash-attn "
                "backend is available on this device (checked: CUDA flash_attn, NPU flash_attn, "
                "HF Hub kernels fallback). Set use_remove_padding=False and ulysses_sp_size=1 to "
                "use the model's native attention implementation instead."
            )

        module._custom_flash_attention_forward = _unavailable_forward
        return

    from transformers.utils import is_flash_attn_greater_or_equal_2_10

    flash_attn_func, flash_attn_varlen_func = fallback
    module.flash_attn_func = flash_attn_func
    module.flash_attn_varlen_func = flash_attn_varlen_func
    module._flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    module._flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    module._flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    logger.info(f"[verl_hardware_plugin] Bound {module.__name__} flash-attn dispatch to HF Hub kernels fallback")


def apply() -> None:
    global _applied
    if _applied:
        return
    if not _xpu_available():
        return

    for module_name in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        _patch_module(module)

    _applied = True
