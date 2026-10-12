# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for the plugin-only Ray device pinning patch.

Mock only -- no accelerator, no Ray cluster. These cover what the patch does
*and* why it depends on import order rather than a documented contract: the
patch is only ever installed ahead of ``Worker.__init__`` by accident, and
"late" is not one failure mode but three, of increasing severity --
``_configure_with_store`` still runs in time (warns, still correct), the
fallback target is itself already running (fixed up directly from the frame,
warns), or the fallback was removed entirely (silently wrong, the worst case).
"""

import os

import pytest

from verl.single_controller.base.worker import Worker
from verl_hardware_plugin.accelerators.xpu.patches import ray_device_pin_patch as patch


class _DeviceRecorder:
    """Stand-in for ``torch.xpu``."""

    def __init__(self):
        self.pinned = []

    def set_device(self, index):
        self.pinned.append(index)


@pytest.fixture
def recorder(monkeypatch):
    rec = _DeviceRecorder()
    monkeypatch.setattr(patch, "_device_module", lambda: rec)
    monkeypatch.setattr(patch, "_xpu_selected", lambda: True)
    monkeypatch.setattr(patch, "_xpu_hardware_present", lambda: True)

    # Start from pristine core. This is not test hygiene for its own sake: on
    # this CPU-only host, with no XPU and "nvidia" the selected platform,
    # ``import verl`` has *already* installed this patch on core's ``Worker``
    # -- platform detection constructs ``PlatformXPU()`` to probe it, and
    # ``PlatformXPU.__init__`` is where the patch is applied. Only the call-time
    # ``_xpu_selected()`` check keeps it inert in a real run.
    patch._reset()
    yield rec
    patch._reset()


@pytest.fixture
def ray_actor(monkeypatch):
    """Fake a Ray actor that holds accelerator ids, plus the worker env."""

    state = {"ids": ["1"], "initialized": True, "runtime_context_calls": 0}

    class _RuntimeContext:
        def get_accelerator_ids(self):
            return {"GPU": state["ids"]} if state["ids"] else {}

    import ray

    def _get_runtime_context():
        state["runtime_context_calls"] += 1
        return _RuntimeContext()

    monkeypatch.setattr(ray, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(ray, "get_runtime_context", _get_runtime_context)
    monkeypatch.setattr("verl.utils.device.get_resource_name", lambda: "GPU")

    for key, val in {
        "WORLD_SIZE": "2",
        "RANK": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "12355",
    }.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    # Ray's default: it masks the actor itself, so no NOSET variable is set.
    for key in list(os.environ):
        if key.startswith("RAY_EXPERIMENTAL_NOSET_"):
            monkeypatch.delenv(key, raising=False)

    return state


def test_core_alone_leaves_every_rank_on_device_zero(ray_actor, recorder):
    """The bug, with no patch applied: Ray assigned physical device 1 and core
    pins nothing, so this rank uses the process default -- device 0."""
    worker = Worker()

    assert recorder.pinned == []
    assert worker._local_rank == 0


def test_patch_pins_the_physical_id_ray_assigned(ray_actor, recorder):
    patch.apply()

    worker = Worker()

    assert recorder.pinned == [1]
    assert os.environ["LOCAL_RANK"] == "1"
    assert worker._local_rank == 1


def test_cpu_only_actor_is_not_pinned(ray_actor, recorder):
    ray_actor["ids"] = []
    patch.apply()

    worker = Worker()

    assert recorder.pinned == []
    assert worker._local_rank == 0


def test_in_process_worker_never_queries_ray(ray_actor, recorder):
    """``Worker`` is also built outside Ray (``SFTTrainer._build_engine()``
    under torchrun). Querying Ray there would start a local Ray instance."""
    ray_actor["initialized"] = False
    patch.apply()

    Worker()

    assert recorder.pinned == []
    assert ray_actor["runtime_context_calls"] == 0


def test_patch_applied_too_late_falls_back_to_private_store_key(ray_actor, recorder, caplog, monkeypatch):
    """Simulate the ordering this patch is not entitled to assume: the platform
    singleton -- and so ``apply()`` -- is first built by the
    ``get_resource_name()`` call *inside*
    ``Worker._setup_env_cuda_visible_devices``, i.e. after the method the patch
    wants has already started."""
    original_setup = Worker._setup_env_cuda_visible_devices

    def _setup_that_builds_the_platform(self):
        patch.apply()  # stands in for get_resource_name() -> PlatformXPU()
        return original_setup(self)

    monkeypatch.setattr(Worker, "_setup_env_cuda_visible_devices", _setup_that_builds_the_platform)

    with caplog.at_level("WARNING"):
        worker = Worker()

    # Still correct, but only because of the second patch target.
    assert recorder.pinned == [1]
    assert worker._local_rank == 1
    assert any("_configure_with_store" in rec.message for rec in caplog.records)


def test_patch_applied_too_late_without_the_fallback_is_silently_wrong(ray_actor, recorder, monkeypatch):
    """What the late case costs if the fallback is not there: patching the
    method that is already running changes nothing for this worker, and nothing
    reports it. Two ranks share a device; the run dies later, out of memory."""
    monkeypatch.setattr(patch, "_worker_init_running", lambda: False)
    original_setup = Worker._setup_env_cuda_visible_devices

    def _setup_that_builds_the_platform(self):
        patch.apply()
        return original_setup(self)

    monkeypatch.setattr(Worker, "_setup_env_cuda_visible_devices", _setup_that_builds_the_platform)

    worker = Worker()

    assert recorder.pinned == []
    assert worker._local_rank == 0


def test_patch_triggered_from_inside_configure_with_store_still_pins_this_worker(
    ray_actor, recorder, caplog, monkeypatch
):
    """Deeper than the fallback above: ``_configure_with_store`` calls
    ``type(self).env_keys()``, which calls ``get_visible_devices_keyword()``,
    which calls ``get_platform()`` -- so the platform singleton, and therefore
    ``apply()``, can be built from *inside* an in-flight ``_configure_with_store``
    call. At that point even the fallback target has already been looked up
    for this call; reinstalling it on the class cannot change what this call
    is already running. This worker must be pinned directly from the frame."""
    original_env_keys = Worker.env_keys.__func__

    def _env_keys_that_builds_the_platform(cls):
        patch.apply()  # stands in for get_visible_devices_keyword() -> PlatformXPU()
        return original_env_keys(cls)

    monkeypatch.setattr(Worker, "env_keys", classmethod(_env_keys_that_builds_the_platform))

    with caplog.at_level("WARNING"):
        worker = Worker()

    assert recorder.pinned == [1]
    assert worker._local_rank == 1


def test_in_flight_fallback_never_resolves_get_platform(ray_actor, recorder, monkeypatch):
    """Regression test: ``get_platform()`` must never be called from the
    in-flight fallback, neither directly (``_xpu_selected()``) nor via
    ``get_resource_name() -> get_platform().ray_resource_name()``. At that
    point ``PlatformXPU`` is still under construction, so the platform
    singleton either call would resolve is the one currently being built on
    the stack above it -- recursing into that same construction and growing
    the stack without bound until ``RecursionError``, which verl's own
    detection loop can swallow via a bare ``except Exception``, silently
    selecting the wrong platform instead of pinning this worker.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("in-flight fallback must not resolve get_platform()")

    monkeypatch.setattr(patch, "_xpu_selected", _must_not_be_called)
    monkeypatch.setattr("verl.utils.device.get_resource_name", _must_not_be_called)

    original_env_keys = Worker.env_keys.__func__

    def _env_keys_that_builds_the_platform(cls):
        patch.apply()
        return original_env_keys(cls)

    monkeypatch.setattr(Worker, "env_keys", classmethod(_env_keys_that_builds_the_platform))

    worker = Worker()

    assert recorder.pinned == [1]
    assert worker._local_rank == 1


def test_import_verl_already_installs_the_patch_on_any_host(tmp_path):
    """Both halves of what makes this a demonstration, in one subprocess.

    1. ``import verl`` alone -- no worker, no Ray -- already replaces core's
       ``Worker._setup_env_cuda_visible_devices`` with this module's function,
       on a host with no XPU at all, because platform detection constructs
       ``PlatformXPU()`` to probe it.
    2. That is also the only reason the patch is ever in time: it has to be
       installed before the first ``Worker`` is built, and nothing in core
       promises that the platform singleton exists by then.
    """
    import subprocess
    import sys

    script = (
        "import verl\n"
        "from verl.single_controller.base.worker import Worker\n"
        "from verl.plugin.platform import get_platform\n"
        "print(get_platform().device_name, Worker._setup_env_cuda_visible_devices.__module__)\n"
    )
    # Entry-point discovery and auto-detection both on, whatever the parent
    # suite's fixtures set for themselves.
    env = {k: v for k, v in os.environ.items() if k not in ("VERL_USE_EXTERNAL_PLUGINS", "VERL_PLATFORM")}

    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.split()

    patched_from = out[-1]
    assert patched_from == "verl_hardware_plugin.accelerators.xpu.patches.ray_device_pin_patch"


def test_apply_is_idempotent(ray_actor, recorder):
    patch.apply()
    patch.apply()

    Worker()

    assert recorder.pinned == [1]
