# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Allow VeOmni's low-precision ReduceScatter transport on XPU.

Delete this module once VeOmni accepts XPU for ``low_precision_reduce_scatter_comm`` (VeOmni#1167 added the
transport with a CUDA-only gate).

``engine.low_precision_reduce_scatter_comm=True`` (with ``engine.mixed_precision=True``) makes FSDP2 send
gradients as BF16 and reduce them in FP32. The transport itself is device-agnostic (one ``all_to_all_single``
plus a local FP32 sum, see ``veomni/distributed/fsdp2/reduce_scatter.py``), but ``parallelize_model_fsdp2``
raises "only supported on CUDA/NCCL" for any other device. That check is a statement inside the function
body, so the function is recompiled from its own source with ``xpu`` allowed and the new code object is
swapped onto the existing function (callers that imported it by name see the change).

Compatible with the plugin's oneCCL force-SUM workaround: the transport already forces SUM and owns the
gradient scaling, and setting force-SUM again is idempotent.
"""

import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)

_RAISE = '\n            raise RuntimeError("Low-precision ReduceScatter transport'
_OLD = 'if get_device_type() != "cuda":' + _RAISE
_NEW = 'if get_device_type() not in ("cuda", "xpu"):' + _RAISE


def apply() -> None:
    try:
        from veomni.distributed import torch_parallelize as tp
    except ImportError:
        return

    fn = tp.parallelize_model_fsdp2
    if getattr(fn, "_xpu_lprs_applied", False):
        return
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return
    if src.count(_OLD) != 1:
        return  # VeOmni already allows xpu, or the code changed; nothing to bridge

    namespace = dict(tp.__dict__)
    exec(compile(src.replace(_OLD, _NEW), tp.__file__, "exec"), namespace)
    fn.__code__ = namespace[fn.__name__].__code__
    fn._xpu_lprs_applied = True
    logger.debug("Allowed VeOmni's low-precision ReduceScatter transport on xpu")
