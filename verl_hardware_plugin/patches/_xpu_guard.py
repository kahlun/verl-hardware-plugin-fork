# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Shared Intel XPU availability check for the patch modules.

Every module in this package mutates global verl-core / torch state, so each
one must be a no-op on a CPU/CUDA/NPU process that happens to import this
plugin (the entry point is discovered on every host, not just Intel ones).
One helper, used by all of them, so the guard cannot drift between modules.
"""


def xpu_available() -> bool:
    """True only when this process actually has a usable Intel XPU device."""
    import torch

    return hasattr(torch, "xpu") and torch.xpu.is_available()
