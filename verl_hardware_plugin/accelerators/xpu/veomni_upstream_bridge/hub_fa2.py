# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bridge for the VeOmni change that runs the Hub FlashAttention2 adapter on XPU.

Delete this module once VeOmni has an ``xpu`` row in ``_FA2_HUB_ROWS`` and resolves the
hub kernel version itself (``_hub_flash_kernel_version``). The clean upstream version of
this change is the VeOmni patch ``veomni_xpu_hub_fa2.patch`` (branch
``feat/xpu-hub-flash-attention``). Needed for Ulysses sequence parallel on XPU, where
stock ``flash_attn`` is unavailable.
"""

import functools


def _use_xpu_hub_flash_kernel_version() -> None:
    """Make VeOmni's Hub FlashAttention loader request the version Transformers selects.

    VeOmni requests a fixed major version of the Hub kernel, which has no XPU build. Resolve the
    version with Transformers' ``get_attn_kernel_version`` instead. No-op if VeOmni already does
    this or if Transformers has no resolver.
    """
    try:
        import veomni.ops.install as veomni_install
        from kernels import get_kernel
        from transformers.integrations.hub_kernels import get_attn_kernel_version
    except ImportError:
        return

    if hasattr(veomni_install, "_hub_flash_kernel_version"):
        return  # VeOmni resolves the hub version itself

    @functools.cache
    def _load_hub_flash_kernel(repository: str):
        return get_kernel(repository, version=get_attn_kernel_version(repository))

    veomni_install._load_hub_flash_kernel = _load_hub_flash_kernel


def _register_xpu_hub_flash_attention() -> None:
    """Register VeOmni's Hub FlashAttention2 attention ops for the ``xpu`` device.

    VeOmni's op registry is keyed by (op, variant, impl, device) and registers the Hub adapter
    for ``cuda`` only. The adapter itself is device-agnostic, so reuse it under an XPU
    requirement. No-op if the ``xpu`` entry already exists.
    """
    try:
        import torch
        from veomni.ops import registry as veomni_registry
    except ImportError:
        return

    class _XpuKernelRequirement:
        device = "xpu"

        def matches(self) -> bool:
            return hasattr(torch, "xpu") and torch.xpu.is_available()

        def check(self) -> None:
            if not self.matches():
                raise RuntimeError("XpuKernelRequirement is not satisfied (no XPU device)")

    registry = veomni_registry.OP_REGISTRY
    for impl in ("flash_attention_2_hub", "veomni_flash_attention_2_hub"):
        cuda_entry = registry._entries.get(("attention", "standard", impl, "cuda"))
        if cuda_entry is None or ("attention", "standard", impl, "xpu") in registry._entries:
            continue
        veomni_registry.register_op(
            "attention",
            "standard",
            impl,
            description=f"{cuda_entry.description} (XPU)",
            wrapper=cuda_entry.wrapper,
            requirement=_XpuKernelRequirement(),
            requires=("kernels",),
        )


def apply() -> None:
    _use_xpu_hub_flash_kernel_version()
    _register_xpu_hub_flash_attention()
