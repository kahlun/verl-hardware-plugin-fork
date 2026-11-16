# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bridge for VeOmni#1387 (Intel XPU device selection and xccl backend).

Delete this module once verl's pinned VeOmni contains #1387.

Without #1387 ``veomni.utils.device.get_device_type()`` returns ``cpu`` on XPU, so the
model never leaves the CPU. Other VeOmni modules bind these functions by value
(``from veomni.utils.device import get_device_type``), so replacing the module attribute
would miss them. Instead the function *code* is swapped in place: every existing
reference sees the new behavior, whenever VeOmni was imported. The replacement bodies run
with ``veomni.utils.device``'s own globals, so ``IS_XPU_AVAILABLE`` is set there too.

Mirrors the ``device.py`` hunks of #1387 exactly, plus the ``helper.empty_cache`` hunk.
"""

import logging

logger = logging.getLogger(__name__)


# The three functions below are only code donors: their bodies are copied onto the VeOmni
# functions. Names they use (IS_*_AVAILABLE, torch, synchronize) resolve in VeOmni's module.
def _get_device_type() -> str:
    if IS_CUDA_AVAILABLE:  # noqa: F821
        device = "cuda"
    elif IS_NPU_AVAILABLE:  # noqa: F821
        device = "npu"
    elif IS_MLU_AVAILABLE:  # noqa: F821
        device = "mlu"
    elif IS_XPU_AVAILABLE:  # noqa: F821
        device = "xpu"
    else:
        device = "cpu"

    return device


def _get_dist_comm_backend() -> str:
    if IS_CUDA_AVAILABLE:  # noqa: F821
        return "nccl"
    elif IS_NPU_AVAILABLE:  # noqa: F821
        return "hccl"
    elif IS_MLU_AVAILABLE:  # noqa: F821
        return "cncl"
    elif IS_XPU_AVAILABLE:  # noqa: F821
        return "xccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")  # noqa: F821


def _stream_synchronize() -> None:
    if IS_CUDA_AVAILABLE:  # noqa: F821
        torch.cuda.current_stream().synchronize()  # noqa: F821
    elif IS_NPU_AVAILABLE:  # noqa: F821
        torch.npu.current_stream().synchronize()  # noqa: F821
    elif IS_MLU_AVAILABLE:  # noqa: F821
        torch.mlu.current_stream().synchronize()  # noqa: F821
    elif IS_XPU_AVAILABLE:  # noqa: F821
        torch.xpu.current_stream().synchronize()  # noqa: F821
    else:
        synchronize()  # noqa: F821


def _helper_empty_cache() -> None:
    # Uses helper.py's globals (gc, IS_*_AVAILABLE).
    gc.collect()  # noqa: F821

    if IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE or IS_MLU_AVAILABLE or get_device_type() == "xpu":  # noqa: F821
        from veomni.utils.device import empty_cache

        empty_cache()


def apply() -> None:
    try:
        import torch
        from veomni.utils import device as veomni_device
        from veomni.utils import helper as veomni_helper
    except ImportError:
        return

    if hasattr(veomni_device, "IS_XPU_AVAILABLE"):
        return  # VeOmni has #1387 (or the bridge already ran)

    veomni_device.IS_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    for name, donor in (
        ("get_device_type", _get_device_type),
        ("get_dist_comm_backend", _get_dist_comm_backend),
        ("stream_synchronize", _stream_synchronize),
    ):
        getattr(veomni_device, name).__code__ = donor.__code__
    veomni_helper.empty_cache.__code__ = _helper_empty_cache.__code__
    logger.debug("Applied VeOmni#1387 device bridge (xpu=%s)", veomni_device.IS_XPU_AVAILABLE)
