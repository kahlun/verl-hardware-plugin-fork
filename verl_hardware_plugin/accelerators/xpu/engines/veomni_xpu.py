# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""VeOmni engine for Intel XPU devices.

verl-core registers its VeOmni engine for ``device=["cuda", "npu"]`` only, so on
XPU ``EngineRegistry.get_engine_cls(model_type, "veomni")`` finds nothing. These
subclasses register the same engine under ``(xpu, intel)`` and add the same
oneCCL workaround the XPU FSDP engine applies (see ``fsdp_xpu.py``): VeOmni's
FSDP2 wrappers also default to ReduceOp.AVG, which oneCCL can abort on, so
force SUM + manual division on every FSDP2 module after initialization.

No core code is patched; this is ordinary subclassing. Gaps in VeOmni itself (device
selection, hub FA2 on XPU) are bridged by ``../veomni_upstream_bridge/``, which is meant to be
deleted once VeOmni supports XPU upstream.
"""

import logging
import os

from verl_hardware_plugin.accelerators.xpu import veomni_upstream_bridge

# Temporary: lets the engine run on an unmodified VeOmni. Delete with the folder (see its README).
veomni_upstream_bridge.apply()

from verl.workers.engine.base import EngineRegistry  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead, VeOmniEngineWithValueHead  # noqa: E402

from .fsdp_xpu import _force_sum_reduction_on_all_fsdp_modules  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _release_xpu_cache() -> None:
    """Release cached, unused device memory back to the driver.

    Called after each engine pass so the allocator's cache does not accumulate across phases
    and starve colocated processes such as the rollout engine. Set VERL_XPU_EMPTY_CACHE=0 to disable.
    """
    if os.getenv("VERL_XPU_EMPTY_CACHE", "1") == "0":
        return
    import torch

    torch.xpu.empty_cache()


def _trim_threshold_bytes() -> float:
    """Reserved-pool size above which the cache is trimmed after a micro-batch.

    VERL_XPU_EMPTY_CACHE_RESERVED_GB sets it in GiB (0 trims after every micro-batch). Otherwise it is
    VERL_XPU_EMPTY_CACHE_RESERVED_FRACTION (default 1/3) of the card: 8 GiB on a 24 GiB B60, where the
    allocator was seen to fill the card, and about 50 GiB on a 152 GiB CRI card. A fixed 8 GiB on CRI
    trimmed after nearly every micro-batch and made VeOmni GRPO actor updates about 1.7x slower.
    """
    gb = os.getenv("VERL_XPU_EMPTY_CACHE_RESERVED_GB")
    if gb is not None:
        return float(gb) * 1024**3
    import torch

    fraction = float(os.getenv("VERL_XPU_EMPTY_CACHE_RESERVED_FRACTION", str(1 / 3)))
    return fraction * torch.xpu.get_device_properties(torch.xpu.current_device()).total_memory


def _release_xpu_cache_if_bloated() -> None:
    """Release cached device memory once the allocator's reserved pool exceeds a threshold.

    Batches with variable sequence lengths can make the caching allocator reserve new segments
    instead of reusing cached ones. The XPU runtime may fail an allocation without first freeing
    the cache and retrying, so the pool is trimmed proactively. Threshold: see _trim_threshold_bytes.
    Set VERL_XPU_EMPTY_CACHE=0 to disable.
    """
    if os.getenv("VERL_XPU_EMPTY_CACHE", "1") == "0":
        return
    import torch

    if torch.xpu.memory_reserved() > _trim_threshold_bytes():
        torch.xpu.empty_cache()


def _cap_xpu_memory_fraction() -> None:
    """Cap this process's XPU caching allocator at a fraction of the card (opt-in).

    The xe driver never fails an over-commit, so the allocator never takes its free-cached-and-retry
    path: reserved memory grows with each distinct micro-batch length until Level Zero raises
    UR_RESULT_ERROR_OUT_OF_RESOURCES. A per-process fraction makes the allocator release cached
    blocks at the cap. Alternative to the cache trimming above, from the CRI Qwen3.5 GRPO runs
    (not yet compared against it on VeOmni). Set VERL_XPU_MEM_FRACTION=0.85 to enable; unset = off.
    """
    value = os.getenv("VERL_XPU_MEM_FRACTION")
    if not value or float(value) <= 0:
        return
    import torch

    torch.xpu.set_per_process_memory_fraction(float(value))


def _log_xpu_memory(tag: str) -> None:
    """Print device memory usage when VERL_XPU_MEMLOG=1 (debugging aid)."""
    if os.getenv("VERL_XPU_MEMLOG") != "1":
        return
    import torch

    gb = 1024**3
    free, total = torch.xpu.mem_get_info()
    print(
        f"[xpu-mem] pid={os.getpid()} dev={torch.xpu.current_device()} {tag}: "
        f"alloc={torch.xpu.memory_allocated() / gb:.2f}G "
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
        _cap_xpu_memory_fraction()
        super().initialize()
        _force_sum_reduction_on_all_fsdp_modules(self.module)


@EngineRegistry.register(model_type="value_model", backend=["veomni"], device="xpu", vendor="intel")
class VeOmniXPUEngineWithValueHead(VeOmniEngineWithValueHead):
    """VeOmni value-model engine for Intel XPU with xccl communication backend."""

    def initialize(self):
        super().initialize()
        _force_sum_reduction_on_all_fsdp_modules(self.module)
