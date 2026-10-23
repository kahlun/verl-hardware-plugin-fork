# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Offer ``W4A16`` (weight-only INT4) as a ``rollout.quantization`` preset on Intel XPU.

verl's low-precision rollout (``rollout.quantization.linear`` / ``moe``) accepts the
compressed-tensors presets listed in ``verl.utils.qat.compressed_tensors.SUPPORTED_SCHEMES``:
the ones whose refit path verl has validated. Everything the preset needs already exists in
core -- the trainer-side exporter packs group-strategy weights with compressed-tensors'
``pack-quantized`` compressor -- so the only missing piece is the name in that table.

On Intel XPU the matching engine kernel is vLLM's ``XPUwNa16LinearKernel``
(``torch.ops._xpu_C.int4_gemm_w4a16``), reached through the engine's ordinary
compressed-tensors W4A16 scheme. The preset has no QAT mode (``None``): it is post-training
round-to-nearest of the trainer's weights at every refit.

How it is applied: ``SUPPORTED_SCHEMES`` is a module-level dict read at call time, so adding
a key is enough. The module may not be imported yet when the platform is constructed
(``PlatformXPU()`` runs while ``import verl`` is still in progress), so if it is absent a
one-shot ``sys.meta_path`` hook adds the key right after its first import instead of
importing it early.
"""

import importlib.abc
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TARGET = "verl.utils.qat.compressed_tensors"
PRESET = "W4A16"
_applied = False


def _xpu_available() -> bool:
    import torch

    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _register(module) -> None:
    module.SUPPORTED_SCHEMES.setdefault(PRESET, None)
    logger.info("[verl_hardware_plugin] Registered rollout quantization preset %s", PRESET)


class _AfterImport(importlib.abc.MetaPathFinder):
    """Run ``_register`` once, right after ``_TARGET`` finishes importing."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)  # one-shot; also lets the real finders resolve the module below
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_then_register(module):
            exec_module(module)
            _register(module)

        loader.exec_module = exec_then_register
        return spec


def apply() -> None:
    global _applied
    if _applied:
        return
    if not _xpu_available():
        return
    _applied = True
    module = sys.modules.get(_TARGET)
    if module is not None:
        _register(module)
    else:
        sys.meta_path.insert(0, _AfterImport())


def _reset() -> None:
    """Test hook: forget that the patch was applied and drop a pending import hook."""
    global _applied
    _applied = False
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, _AfterImport)]
