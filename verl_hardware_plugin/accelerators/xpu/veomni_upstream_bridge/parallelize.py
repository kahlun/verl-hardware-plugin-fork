# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bridge for the single-process (no FSDP) path of VeOmni#1387.

Delete this module once verl's pinned VeOmni contains #1387.

Without FSDP (world size 1), ``build_parallelize_model`` only accepts ``init_device`` in an inline
list ``["cuda", "npu", "mlu"]`` and raises "Only FSDP training supports ``init_device=meta``" for
``xpu``. #1387 adds ``xpu`` to that list. The list is a literal inside the function body, so the
function is recompiled from its own source with the literal widened, and the new code object is
swapped onto the existing function (callers that imported it by name see the change).
"""

import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)

_OLD = '["cuda", "npu", "mlu"]'
_NEW = '["cuda", "npu", "mlu", "xpu"]'


def apply() -> None:
    try:
        from veomni.distributed import torch_parallelize as tp
    except ImportError:
        return

    fn = tp.build_parallelize_model
    if getattr(fn, "_xpu_bridge_applied", False):
        return
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return
    if _OLD not in src:
        return  # VeOmni already accepts xpu (or the code changed); nothing to bridge

    namespace = dict(tp.__dict__)
    exec(compile(src.replace(_OLD, _NEW), tp.__file__, "exec"), namespace)
    fn.__code__ = namespace[fn.__name__].__code__
    fn._xpu_bridge_applied = True
    logger.debug("Applied VeOmni#1387 init_device bridge for the non-FSDP path")
