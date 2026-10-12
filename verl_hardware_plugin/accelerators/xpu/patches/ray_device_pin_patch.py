# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Pin each Ray-launched XPU worker to the physical device Ray assigned it.

``Worker._setup_env_cuda_visible_devices()`` only calls ``set_device()`` when
a ``RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`` env var is set. By default verl
relies on Ray's per-actor visibility mask to make index 0 the correct device.
torch-xpu discards ``ZE_AFFINITY_MASK`` after lazy init, so every actor
enumerates all physical devices and ``xpu:N`` resolves to physical device N
regardless of the mask, putting every rank on ``xpu:0``. The collision does
not raise; it surfaces as OOM or as collectives taking a shared-device path.

This module patches ``Worker`` from the plugin side, with no verl-core
change. The core-side alternative is ``PlatformBase.ray_device_index()``,
proposed in verl-project/verl#8145; this module exists to show what plugin-
only pinning costs relative to that hook, not to replace it.

Patch target is selected automatically at install time:

- ``Worker._setup_env_cuda_visible_devices`` (preferred). Wraps the method:
  calls through to core's implementation, then pins the physical device via
  ``ray.get_runtime_context().get_accelerator_ids()``. Installed from
  ``PlatformXPU.__init__``, which runs before any ``Worker`` is constructed
  because this plugin's ``register_all_engines()`` imports a core engine
  module whose module body builds the platform singleton as a side effect.

- ``Worker._configure_with_store`` (fallback). Used when ``apply()`` is
  itself invoked from inside ``Worker.__init__`` (detected via a call-stack
  check), meaning the preferred target already ran for this instance.
  Writes ``store["_local_rank"]`` directly instead.

Both targets are private core API. ``_reset()`` is for tests only.
"""

import logging
import os
import sys
from collections.abc import Callable
from types import FrameType
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False

# Saved so _reset() can restore them. apply() can run from a platform
# detection probe on a host where XPU is not the selected platform, so
# "installed" and "active" are tracked separately via _xpu_selected().
_originals: dict[str, Callable] = {}


def _xpu_selected() -> bool:
    """Whether XPU is the platform verl selected for this process."""
    from verl.plugin.platform import get_platform

    try:
        return get_platform().device_name == "xpu"
    except Exception:  # pragma: no cover - platform not resolvable yet
        return False


def _device_module():
    import torch

    return torch.xpu


def _ray_physical_index():
    """Physical device index Ray assigned to this actor, or None.

    None covers in-process ``Worker`` construction with no Ray (e.g.
    ``SFTTrainer._build_engine()`` under torchrun) and CPU-only actors.
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
    """Whether a ``Worker.__init__`` frame is already on the call stack."""
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
        # Preferred target already ran for this instance. Fix up at the
        # next core entry point in the same __init__: _configure_with_store
        # re-exports store["_local_rank"] to the LOCAL_RANK env var.
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
    """Restore the patched methods. Test-only; a real run never un-patches."""
    global _applied

    from verl.single_controller.base.worker import Worker

    for name, original in _originals.items():
        setattr(Worker, name, original)
    _originals.clear()
    _applied = False
