# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU-only test of the VeOmni#1387 device bridge against a stand-in for stock VeOmni.

Delete together with ``veomni_upstream_bridge/`` once VeOmni supports XPU upstream.
"""

import importlib
import importlib.util
import sys
import types

import pytest
import torch


def _stock_veomni(monkeypatch):
    """Stand-in for VeOmni without #1387: no IS_XPU_AVAILABLE, three call sites bind by value."""
    device = types.ModuleType("veomni.utils.device")
    device.torch = torch
    device.IS_CUDA_AVAILABLE = device.IS_NPU_AVAILABLE = device.IS_MLU_AVAILABLE = False
    exec(
        "def get_device_type():\n"
        "    if IS_CUDA_AVAILABLE: return 'cuda'\n"
        "    elif IS_NPU_AVAILABLE: return 'npu'\n"
        "    elif IS_MLU_AVAILABLE: return 'mlu'\n"
        "    return 'cpu'\n"
        "def get_dist_comm_backend():\n"
        "    if IS_CUDA_AVAILABLE: return 'nccl'\n"
        "    raise RuntimeError('none')\n"
        "def synchronize(): pass\n"
        "def stream_synchronize():\n"
        "    if IS_CUDA_AVAILABLE: torch.cuda.current_stream().synchronize()\n"
        "    else: synchronize()\n",
        device.__dict__,
    )
    helper = types.ModuleType("veomni.utils.helper")
    pkg = types.ModuleType("veomni")
    utils = types.ModuleType("veomni.utils")
    utils.device, utils.helper = device, helper
    for name, mod in (
        ("veomni", pkg),
        ("veomni.utils", utils),
        ("veomni.utils.device", device),
        ("veomni.utils.helper", helper),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    exec(
        "import gc\n"
        "from veomni.utils.device import get_device_type\n"
        "IS_CUDA_AVAILABLE = IS_NPU_AVAILABLE = IS_MLU_AVAILABLE = False\n"
        "def empty_cache():\n"
        "    gc.collect()\n"
        "    if IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE or IS_MLU_AVAILABLE:\n"
        "        pass\n",
        helper.__dict__,
    )
    device.empty_cache = lambda: calls.append("device.empty_cache")
    return device, helper


calls: list = []


def test_bridge_makes_stock_veomni_see_xpu(monkeypatch):
    calls.clear()
    monkeypatch.setattr(torch, "xpu", types.SimpleNamespace(is_available=lambda: True), raising=False)
    device, helper = _stock_veomni(monkeypatch)
    by_value = device.get_device_type  # a call site that bound the function by value
    assert by_value() == "cpu"

    from verl_hardware_plugin.accelerators.xpu.veomni_upstream_bridge import device as bridge

    importlib.reload(bridge)
    bridge.apply()

    assert by_value() == "xpu"  # existing by-value references are updated
    assert device.get_dist_comm_backend() == "xccl"
    helper.empty_cache()
    assert calls == ["device.empty_cache"]
    bridge.apply()  # idempotent


def test_bridge_is_noop_once_veomni_has_xpu(monkeypatch):
    device, _ = _stock_veomni(monkeypatch)
    device.IS_XPU_AVAILABLE = False
    code = device.get_device_type.__code__

    from verl_hardware_plugin.accelerators.xpu.veomni_upstream_bridge import device as bridge

    bridge.apply()
    assert device.get_device_type.__code__ is code


def test_bridge_skipped_when_veomni_absent(monkeypatch):
    for name in [n for n in sys.modules if n == "veomni" or n.startswith("veomni.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [m for m in sys.meta_path if "veomni" not in repr(m)])
    from verl_hardware_plugin.accelerators.xpu.veomni_upstream_bridge import device as bridge

    bridge.apply()  # must not raise


def test_low_precision_rs_bridge_allows_xpu(monkeypatch, tmp_path):
    """The CUDA-only gate is a statement in parallelize_model_fsdp2; the bridge recompiles it to allow xpu."""
    src = (
        "def get_device_type():\n"
        "    return DEVICE\n"
        "\n"
        "\n"
        "def parallelize_model_fsdp2(use_low_precision_transport):\n"
        "    if use_low_precision_transport:\n"
        '        if get_device_type() != "cuda":\n'
        '            raise RuntimeError("Low-precision ReduceScatter transport is only supported on CUDA/NCCL.")\n'
        '    return "sharded"\n'
    )
    path = tmp_path / "torch_parallelize.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("veomni.distributed.torch_parallelize", path)
    tp = importlib.util.module_from_spec(spec)
    tp.DEVICE = "xpu"
    spec.loader.exec_module(tp)
    dist = types.ModuleType("veomni.distributed")
    dist.torch_parallelize = tp
    for name, mod in (
        ("veomni", types.ModuleType("veomni")),
        ("veomni.distributed", dist),
        ("veomni.distributed.torch_parallelize", tp),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    by_name = tp.parallelize_model_fsdp2  # build_parallelize_model calls it through the module global
    with pytest.raises(RuntimeError):
        by_name(True)

    from verl_hardware_plugin.accelerators.xpu.veomni_upstream_bridge import low_precision_rs as bridge

    importlib.reload(bridge)
    bridge.apply()
    assert by_name(True) == "sharded"
    tp.DEVICE = "npu"
    with pytest.raises(RuntimeError):
        by_name(True)  # other devices are still rejected
    bridge.apply()  # idempotent
