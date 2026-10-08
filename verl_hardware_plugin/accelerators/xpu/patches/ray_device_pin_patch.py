# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch verl's Ray worker device pinning so each XPU rank lands on its
own physical device, with no verl-core change.

Why this is needed: ``Worker._setup_env_cuda_visible_devices()`` only calls
``set_device()`` when a ``RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`` variable is
set. In the default configuration verl never pins at all and relies on Ray's
per-actor visibility mask relabeling devices so that index 0 is this rank's
device. torch-xpu discards ``ZE_AFFINITY_MASK`` once its lazy init runs, so
every actor still enumerates all devices and ``xpu:N`` is physical device N
regardless of the mask -- which puts every rank of the job on ``xpu:0``. The
collision raises no device error: it surfaces as out-of-memory, or as
collectives taking paths meant for ranks that share a device.

This patch exists to answer one question that came up on
verl-project/verl#8145 -- *can a plugin fix this without touching core?* -- and
the answer it demonstrates is "yes, but only by depending on things a plugin is
not entitled to depend on". It is a demonstration, not a recommendation; see
``docs/design/XPU_RAY_DEVICE_PIN_PLUGIN_ONLY.md`` for the full comparison
against the core hook.

What makes this one materially hackier than this package's other patches:

- **It must win a race that core owns.** The function to patch is the *first
  statement* of ``Worker.__init__``, so the patch has to be installed before
  any worker is constructed. That happens today only because the platform
  singleton -- and so ``PlatformXPU.__init__``, and so this ``apply()`` -- is
  already built during ``import verl``: this plugin's own
  ``register_all_engines()`` imports ``verl.workers.engine.base``, which pulls
  in ``verl/workers/engine/fsdp/transformer_impl.py``, whose module body
  evaluates ``device_name = get_device_name()``. None of that is a contract.
  Make that module-level call lazy in core, or register this plugin's engines
  lazily, and ``PlatformXPU()`` is first constructed by the
  ``get_resource_name()`` call *inside* the very method this patches. The patch
  then lands after the method has already run -- and the failure mode of
  arriving late is not an exception, it is two ranks quietly sharing a device.
  ``apply()`` detects that case (by inspecting the call stack, which is itself
  the kind of thing a plugin should not have to do) and falls back to a second
  patch target, ``Worker._configure_with_store``, which runs later in the same
  ``__init__``.
- **Two private core methods, not one.** Neither ``_setup_env_cuda_visible_
  devices`` nor ``_configure_with_store`` is public API; the fallback path also
  has to know that ``__init__`` reads ``LOCAL_RANK`` from the environment into
  ``store["_local_rank"]`` between the two, so the fixup has to rewrite the
  store dict's private key as well as the environment.
- **It cannot be narrowed to "this run is XPU".** ``_detect_platform_name()``
  instantiates each registered platform in turn to probe it, so
  ``PlatformXPU.__init__`` can also run on a host that merely *has* XPU
  present while a different platform is selected -- and unlike the AVG patch,
  what gets changed here is core's device-pinning decision for whatever
  platform did win. ``apply()`` therefore re-checks the selected platform at
  *call* time rather than at patch time, which means the patch stays installed
  on ``Worker`` for the life of the process and decides per call whether to act.

The clean version of all of this is one 12-line ``PlatformBase.ray_device_
index()`` hook in core (verl-project/verl#8145) plus a 4-line override in
``PlatformXPU``: core keeps the policy "resolve the index from Ray's own
assignment", the plugin keeps the hardware knowledge "our runtime ignores its
own mask", and nothing depends on import order.
"""

import logging
import os
import sys
from types import FrameType
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False

# The core methods this patch replaces, kept so ``_reset()`` can put them back.
# Needed because ``apply()`` runs from ``PlatformXPU.__init__``, which platform
# *detection* also calls on hosts where XPU is not the selected platform, so
# "installed" and "active" are not the same thing here.
_originals = {}


def _xpu_selected() -> bool:
    """True when XPU is the platform verl actually selected for this process.

    Checked at call time, not at patch time: ``apply()`` can run from a
    detection probe that constructs ``PlatformXPU()`` on a host where another
    platform ends up winning.
    """
    from verl.plugin.platform import get_platform

    try:
        return get_platform().device_name == "xpu"
    except Exception:  # pragma: no cover - platform not resolvable yet
        return False


def _device_module():
    import torch

    return torch.xpu


def _ray_physical_index():
    """Return the physical device index Ray assigned to this actor, or None.

    None means "nothing to pin": no Ray (in-process ``Worker`` construction
    under torchrun, e.g. ``SFTTrainer._build_engine()``), or a CPU-only actor
    that holds no accelerator of this resource type.
    """
    import ray

    if not ray.is_initialized():
        return None

    from verl.utils.device import get_resource_name

    accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get(get_resource_name())
    if not accelerator_ids:
        return None
    return int(accelerator_ids[0])


def _worker_init_running() -> bool:
    """True when a ``Worker.__init__`` frame is already on the stack.

    If it is, this ``apply()`` was triggered from inside the method it wants to
    patch -- via ``get_resource_name()`` -> ``get_platform()`` ->
    ``PlatformXPU()`` -> here -- and patching that method now is a no-op for
    the worker currently being built.
    """
    from verl.single_controller.base.worker import Worker

    target = Worker.__init__.__code__
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None:
        if frame.f_code is target:
            return True
        frame = frame.f_back
    return False


def apply() -> None:
    global _applied
    if _applied:
        return

    from verl.single_controller.base.worker import Worker

    late = _worker_init_running()

    if not late:
        original_setup = Worker._setup_env_cuda_visible_devices

        def _patched_setup(self):
            # Let core do its ROCR/HIP/CUDA normalization and, in NOSET mode,
            # its own pin -- which resolves to the same physical id this patch
            # would choose, so applying both is harmless.
            original_setup(self)
            if not _xpu_selected():
                return
            index = _ray_physical_index()
            if index is None:
                return
            os.environ["LOCAL_RANK"] = str(index)
            _device_module().set_device(index)
            logger.info("[verl_hardware_plugin] Pinned xpu:%d from Ray's accelerator assignment", index)

        Worker._setup_env_cuda_visible_devices = _patched_setup
        _originals["_setup_env_cuda_visible_devices"] = original_setup
    else:
        # Arrived too late for the method above. Fix up at the next core entry
        # point inside the same ``Worker.__init__``: ``_configure_with_store``,
        # which re-exports ``store["_local_rank"]`` to ``LOCAL_RANK``. Both the
        # store key and the method are private to core.
        original_configure = Worker._configure_with_store

        def _patched_configure(self, store: dict):
            if _xpu_selected():
                index = _ray_physical_index()
                if index is not None:
                    store["_local_rank"] = index
                    _device_module().set_device(index)
                    logger.info(
                        "[verl_hardware_plugin] Late-pinned xpu:%d from Ray's accelerator assignment "
                        "(patch was installed after _setup_env_cuda_visible_devices had run)",
                        index,
                    )
            return original_configure(self, store=store)

        Worker._configure_with_store = _patched_configure
        _originals["_configure_with_store"] = original_configure
        logger.warning(
            "[verl_hardware_plugin] Ray device pinning patched at _configure_with_store: this plugin was "
            "loaded after Worker.__init__ had already started, so the preferred patch target had already "
            "run. Device pinning now depends on core's private store keys."
        )

    _applied = True


def _reset() -> None:
    """Restore the core methods and forget that the patch was applied.

    Only tests need this -- a real run never un-patches. It exists because a
    plugin that patches a core class cannot otherwise undo the patch it
    installed from a detection probe.
    """
    global _applied

    from verl.single_controller.base.worker import Worker

    for name, original in _originals.items():
        setattr(Worker, name, original)
    _originals.clear()
    _applied = False
