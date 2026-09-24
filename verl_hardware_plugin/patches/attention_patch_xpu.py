# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch to route Intel XPU through the NPU flash-attention shim for
rmpad attention padding, with zero verl-core changes.

Dispatch shape that makes this safe (verl/utils/attention_utils.py):
    verl.utils.attention_utils.index_first_axis/pad_input/rearrange/unpad_input
    are stable public wrappers. Each one calls the private
    `_get_attention_functions()` fresh on *every* invocation -- it is not
    cached at import time. That means patching `_get_attention_functions`
    itself takes effect for every caller regardless of whether that caller
    did `from verl.utils.attention_utils import pad_input` before or after
    this patch runs. This is the one hook where import order genuinely does
    not matter.
"""

import logging
import os

from ._xpu_guard import xpu_available

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    if not xpu_available():
        return

    import verl.utils.attention_utils as attention_utils

    original_get_attention_functions = attention_utils._get_attention_functions

    def _patched_get_attention_functions():
        if xpu_available():
            from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input

            attention_utils._index_first_axis = index_first_axis
            attention_utils._pad_input = pad_input
            attention_utils._rearrange = rearrange
            attention_utils._unpad_input = unpad_input
            return index_first_axis, pad_input, rearrange, unpad_input
        return original_get_attention_functions()

    attention_utils._get_attention_functions = _patched_get_attention_functions
    _applied = True
    logger.info("[verl_hardware_plugin] Patched attention_utils._get_attention_functions for XPU")
