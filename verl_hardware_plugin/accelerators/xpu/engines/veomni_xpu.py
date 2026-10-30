# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""VeOmni engine for Intel XPU devices.

verl-core registers its VeOmni engine for ``device=["cuda", "npu"]`` only, so on
XPU ``EngineRegistry.get_engine_cls(model_type, "veomni")`` finds nothing. These
subclasses register the same engine under ``(xpu, intel)`` and add the same
oneCCL workaround the XPU FSDP engine applies (see ``fsdp_xpu.py``): VeOmni's
FSDP2 wrappers also default to ReduceOp.AVG, which oneCCL can abort on, so
force SUM + manual division on every FSDP2 module after initialization.

No core code is patched; this is ordinary subclassing.
"""

import functools
import importlib
import logging
import os
import sys

try:
    importlib.import_module("veomni.models.checkpoint_tensor_loading")
except ModuleNotFoundError:
    # VeOmni main moved this module to veomni.models.checkpoint.convert (VeOmni#1142,
    # same functions); verl main still imports the old path. Alias it so the unmodified
    # core engine module imports against VeOmni main.
    sys.modules["veomni.models.checkpoint_tensor_loading"] = importlib.import_module("veomni.models.checkpoint.convert")

def _use_xpu_hub_flash_kernel_version() -> None:
    """Point VeOmni's hub FlashAttention loader at a hub major version that has an XPU build.

    VeOmni hardcodes ``get_kernel(repo, version=1)`` for ``veomni_flash_attention_2_hub``;
    kernels-community/flash-attn2 v1/v2 have no build for current torch XPU wheels, while v3
    does (verified on torch 2.14.0+xpu, kernels 0.16.2). Select it with
    ``veomni_flash_attention_2_hub``; override the major version with VERL_XPU_FA2_KERNEL_VERSION.
    """
    try:
        import veomni.ops.install as veomni_install
        from kernels import get_kernel
    except ImportError:
        return

    version = int(os.getenv("VERL_XPU_FA2_KERNEL_VERSION", "3"))

    @functools.lru_cache(maxsize=None)
    def _load_hub_flash_kernel(repository: str):
        return get_kernel(repository, version=version)

    veomni_install._load_hub_flash_kernel = _load_hub_flash_kernel


def _register_xpu_hub_flash_attention() -> None:
    """Register VeOmni's hub FlashAttention2 rows for the ``xpu`` device.

    VeOmni's op registry is keyed by (op, variant, impl, device) and only registers the hub
    FA2 adapter for ``cuda``, so ``veomni_flash_attention_2_hub`` fails to resolve on XPU even
    though the adapter itself is device-agnostic (it calls the hub kernel's varlen functions).
    Re-register the existing wrapper under an XPU requirement.
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


_use_xpu_hub_flash_kernel_version()
_register_xpu_hub_flash_attention()

from verl.workers.engine.base import EngineRegistry  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead, VeOmniEngineWithValueHead  # noqa: E402

from .fsdp_xpu import _force_sum_reduction_on_all_fsdp_modules  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _release_xpu_cache() -> None:
    """Return the allocator's cached-but-unused device memory after each engine pass.

    The VeOmni engine never calls empty_cache() between phases, so on a card shared with
    colocated vLLM the XPU caching allocator keeps growing its reserved pool (measured
    ~20 GB reserved vs ~3 GB allocated on a 24 GB B60) until a later allocation fails with
    UR_RESULT_ERROR_OUT_OF_RESOURCES. Set VERL_XPU_EMPTY_CACHE=0 to disable.
    """
    if os.getenv("VERL_XPU_EMPTY_CACHE", "1") == "0":
        return
    import torch

    torch.xpu.empty_cache()


def _release_xpu_cache_if_bloated() -> None:
    """Per-micro-batch guard: free the allocator cache once it has ballooned.

    Variable sequence lengths make the XPU caching allocator request new, differently
    sized segments instead of reusing cached ones. Within a single log-prob pass the
    reserved pool can fill the whole card (22.96 GB reserved vs 3 GB allocated, 0 GB free
    on a 24 GB B60), and the next allocation then fails with a raw
    UR_RESULT_ERROR_OUT_OF_RESOURCES instead of freeing the cache and retrying.
    Threshold: VERL_XPU_EMPTY_CACHE_RESERVED_GB (default 8; 0 = every micro-batch).
    """
    if os.getenv("VERL_XPU_EMPTY_CACHE", "1") == "0":
        return
    import torch

    limit = float(os.getenv("VERL_XPU_EMPTY_CACHE_RESERVED_GB", "8")) * 1024**3
    if torch.xpu.memory_reserved() > limit:
        torch.xpu.empty_cache()


def _log_xpu_memory(tag: str) -> None:
    """Print device memory when VERL_XPU_MEMLOG=1 (debugging aid for UR_RESULT_ERROR_OUT_OF_RESOURCES)."""
    if os.getenv("VERL_XPU_MEMLOG") != "1":
        return
    import torch

    gb = 1024**3
    free, total = torch.xpu.mem_get_info()
    print(
        f"[xpu-mem] pid={os.getpid()} dev={torch.xpu.current_device()} {tag}: alloc={torch.xpu.memory_allocated() / gb:.2f}G "
        f"reserved={torch.xpu.memory_reserved() / gb:.2f}G max_alloc={torch.xpu.max_memory_allocated() / gb:.2f}G "
        f"dev_free={free / gb:.2f}G/{total / gb:.2f}G",
        flush=True,
    )


@EngineRegistry.register(model_type="language_model", backend=["veomni"], device="xpu", vendor="intel")
class VeOmniXPUEngineWithLMHead(VeOmniEngineWithLMHead):
    """VeOmni LM engine for Intel XPU with xccl communication backend."""

    def forward_step(self, *args, **kwargs):
        try:
            return super().forward_step(*args, **kwargs)
        finally:
            _release_xpu_cache_if_bloated()

    def forward_backward_batch(self, *args, **kwargs):
        _log_xpu_memory("before forward_backward_batch")
        try:
            return super().forward_backward_batch(*args, **kwargs)
        finally:
            _release_xpu_cache()
            _log_xpu_memory("after forward_backward_batch")

    def to(self, *args, **kwargs):
        _log_xpu_memory(f"before to({args})")
        out = super().to(*args, **kwargs)
        _log_xpu_memory(f"after to({args})")
        return out

    def initialize(self):
        super().initialize()
        _force_sum_reduction_on_all_fsdp_modules(self.module)


@EngineRegistry.register(model_type="value_model", backend=["veomni"], device="xpu", vendor="intel")
class VeOmniXPUEngineWithValueHead(VeOmniEngineWithValueHead):
    """VeOmni value-model engine for Intel XPU with xccl communication backend."""

    def initialize(self):
        super().initialize()
        _force_sum_reduction_on_all_fsdp_modules(self.module)
