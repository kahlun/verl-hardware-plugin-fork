# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in ``torch.compile`` of each decoder layer for the VeOmni engine on XPU.

Delete this module once VeOmni's own decoder-block compile (``model.accelerator.torch_compile``) accepts
XPU; today it raises "CUDA-only".

``VERL_XPU_VEOMNI_COMPILE=1`` compiles the forward of every ``*DecoderLayer`` in place after VeOmni has
sharded the model, so FSDP2's all-gather and reshard hooks stay outside the compiled region (the same
approach as VeOmni's ``compile_decoder_blocks``). Shapes are dynamic because verl's batches vary in length.
Works for bf16 and with ``float8.py``; compiled float8 needs gradient checkpointing off.
"""

import logging
import os

logger = logging.getLogger(__name__)

ENV = "VERL_XPU_VEOMNI_COMPILE"


def apply() -> None:
    if os.getenv(ENV, "0") != "1":
        return
    try:
        import verl.workers.engine.veomni.transformer_impl as engine_impl
    except ImportError:
        return
    if getattr(engine_impl, "_xpu_compile_applied", False):
        return

    orig_parallelize = engine_impl.build_parallelize_model

    def build_parallelize_model(*args, **kwargs):
        model = orig_parallelize(*args, **kwargs)
        n = 0
        for module in model.modules():
            if type(module).__name__.endswith("DecoderLayer"):
                module.compile(dynamic=True)
                n += 1
        logger.warning("torch.compile: compiled %d decoder layers", n)
        return model

    engine_impl.build_parallelize_model = build_parallelize_model
    engine_impl._xpu_compile_applied = True
