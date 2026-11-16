# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in torchao float8 training for the VeOmni engine on XPU (stand-in for VeOmni#1364).

Delete this module once VeOmni's float8 support (#1364 and follow-ups) accepts XPU. Today #1364
requires an NVIDIA SM89+ device and rejects ``torch.compile``.

Enable with ``VERL_XPU_VEOMNI_FLOAT8=tensorwise`` (or ``rowwise``). Off by default.
``VERL_XPU_VEOMNI_FLOAT8_ALLGATHER=1`` (tensorwise only) also all-gathers the FSDP2 weight shards in float8
and refreshes their scales after every optimizer step. Combine with ``compile_layers.py``
(``VERL_XPU_VEOMNI_COMPILE=1``) for the compiled path; that needs gradient checkpointing off, because
compiled float8 under activation checkpointing raises ``CheckpointError``.

Status (CRI, Qwen2.5-1.5B, 2 ranks FSDP2): float8 runs and tracks the bf16 loss but is slower than bf16,
eager and compiled. torchao float8 only pays off compiled with a large microbatch (see the CRI FP8 report).

What this module sets up around torchao:

1. ``convert_to_float8_training`` runs on the model VeOmni builds, before FSDP2 wraps it.
2. VeOmni's fused SwiGLU MLP reads ``gate/up/down_proj.weight`` directly when the projections are
   ``isinstance(..., nn.Linear)``. ``Float8Linear`` is a subclass, so the MLP would bypass its float8
   forward and silently stay bf16. The plain-projection check is swapped to require the exact type
   (VeOmni#1364 makes the same change).
3. The float8 GEMM needs its inner dimension divisible by 16; in the weight-gradient GEMM that is the token
   count, which verl's packed variable-length batches do not round. torchao's ``pad_inner_dim`` pads it (on CRI
   this gave the same losses, bit for bit, as padding the tokens by hand).
"""

import logging
import os

logger = logging.getLogger(__name__)

ENV = "VERL_XPU_VEOMNI_FLOAT8"
ALLGATHER_ENV = "VERL_XPU_VEOMNI_FLOAT8_ALLGATHER"
_CALLS = [0]  # Float8Linear forward calls, for verifying float8 actually executes


def _plain_projections_donor(mlp):  # code donor: `nn` resolves in veomni's swiglu_mlp module globals
    return all(type(getattr(mlp, name)) is nn.Linear for name in ("gate_proj", "up_proj", "down_proj"))  # noqa: F821


def calls() -> int:
    """Number of Float8Linear forward calls so far in this process."""
    return _CALLS[0]


def _count_calls(float8_linear_cls) -> None:
    if getattr(float8_linear_cls, "_xpu_counted", False):
        return
    orig_forward = float8_linear_cls.forward

    def forward(self, x):
        _CALLS[0] += 1
        return orig_forward(self, x)

    float8_linear_cls.forward = forward
    float8_linear_cls._xpu_counted = True


def _convert(model, recipe: str, all_gather: bool):
    import dataclasses

    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.float8_linear import Float8Linear
    from veomni.ops.kernels import swiglu_mlp

    swiglu_mlp.has_plain_linear_projections.__code__ = _plain_projections_donor.__code__
    _count_calls(Float8Linear)
    config = dataclasses.replace(
        Float8LinearConfig.from_recipe_name(recipe),
        enable_fsdp_float8_all_gather=all_gather,
        pad_inner_dim=True,
    )
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=lambda mod, fqn: (
            "lm_head" not in fqn and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        ),
    )
    n = sum(type(m).__name__ == "Float8Linear" for m in model.modules())
    logger.warning(
        "float8 training: converted %d Linear -> Float8Linear (recipe=%s, fsdp float8 all-gather=%s)",
        n,
        recipe,
        all_gather,
    )
    return model


def apply() -> None:
    recipe = os.getenv(ENV, "")
    if not recipe:
        return
    try:
        import verl.workers.engine.veomni.transformer_impl as engine_impl
    except ImportError:
        return
    if getattr(engine_impl, "_xpu_float8_applied", False):
        return

    all_gather = os.getenv(ALLGATHER_ENV, "0") == "1"
    if all_gather and recipe != "tensorwise":
        raise ValueError(f"{ALLGATHER_ENV}=1 needs the tensorwise recipe, got {recipe!r}")
    orig_build = engine_impl.build_foundation_model

    def build_foundation_model(*args, **kwargs):
        return _convert(orig_build(*args, **kwargs), recipe, all_gather)

    engine_impl.build_foundation_model = build_foundation_model

    orig_step = engine_impl.VeOmniEngine.optimizer_step

    def optimizer_step(self, *args, **kwargs):
        out = orig_step(self, *args, **kwargs)
        if all_gather:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(self.module)
        if not getattr(self, "_xpu_float8_logged", False):
            # proof that the float8 GEMMs actually ran (the MLP bypass above would leave this near zero)
            logger.warning("float8 training: %d Float8Linear forward calls before the first optimizer step", calls())
            self._xpu_float8_logged = True
        return out

    engine_impl.VeOmniEngine.optimizer_step = optimizer_step
    engine_impl._xpu_float8_applied = True
