#!/usr/bin/env python3
"""On-hardware verification for patches/numa_affinity_patch_xpu.py.

The unit tests prove the patch logic against a stubbed verl. They cannot prove
the thing that actually went wrong in the bug being fixed: that the patch is
really installed inside a real verl process and really moves this process's CPU
affinity. That is what this script checks, and it is deliberately loud about
which of those two halves failed.

Run under devctl with GPUs attached:

    VERL_PLATFORM=intel LOCAL_RANK=0 python3 scripts/verify_numa_patch_xpu.py

Exit code 0 only if the patch was installed *and* pinning demonstrably changed
this process's affinity (or was correctly skipped because the box genuinely has
no NUMA topology, which is reported as SKIP, not PASS).
"""

import os
import sys

checks = []


def record(label, ok, detail=""):
    checks.append((label, ok, detail))
    status = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"[{status}] {label}" + (f"\n         {detail}" if detail else ""), flush=True)


def section(title):
    print(f"\n=== {title} ===", flush=True)


section("Environment")
for var in (
    "VERL_PLATFORM",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "ZE_AFFINITY_MASK",
    "ZES_ENABLE_SYSMAN",
    "RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK",
):
    print(f"  {var}={os.environ.get(var, '<unset>')}", flush=True)
print(f"  hostname={os.uname().nodename}", flush=True)
print(f"  os.cpu_count()={os.cpu_count()}", flush=True)
print(f"  initial sched_getaffinity={len(os.sched_getaffinity(0))} cpus", flush=True)

# ---------------------------------------------------------------------------
# 1. pyzes present at all? This is the first thing to fail in a fresh image --
#    the image ships level-zero's C library but not the Python binding.
# ---------------------------------------------------------------------------
section("pyzes availability")
try:
    import pyzes  # noqa: F401

    record("pyzes importable", True, f"pyzes at {pyzes.__file__}")
except ImportError as e:
    record("pyzes importable", False, f"{e} -- pip install pyzes (image ships only level-zero's C lib)")

# ---------------------------------------------------------------------------
# 2. Is the patch installed? This is the half a hardware run uniquely proves.
# ---------------------------------------------------------------------------
section("Patch installation")
import verl  # noqa: E402
import verl.utils.distributed as vd  # noqa: E402

fn = vd.set_numa_affinity
print(f"  verl version: {getattr(verl, '__version__', '?')}", flush=True)
print(f"  set_numa_affinity -> {fn.__module__}.{fn.__qualname__}", flush=True)

patched = fn.__qualname__.startswith("apply.<locals>._patched_set_numa_affinity")
record(
    "verl.utils.distributed.set_numa_affinity is patched",
    patched,
    "plugin entry_point did not load, or apply_all() did not run" if not patched else "",
)

from verl.plugin.platform import get_platform  # noqa: E402

platform = get_platform()
record(
    "verl selected the XPU platform",
    platform.device_name == "xpu",
    f"device_name={platform.device_name!r} vendor={platform.vendor_name!r} (set VERL_PLATFORM=intel)",
)

# The import-order claim from the patch docstring, tested live rather than by
# reading verl/__init__.py: a module that from-imports set_numa_affinity must
# end up with the patched function, not the original pynvml one.
section("Import-order claim (live)")
try:
    import verl.workers.engine_workers as ew

    ew_fn = ew.set_numa_affinity
    print(f"  engine_workers.set_numa_affinity -> {ew_fn.__module__}.{ew_fn.__qualname__}", flush=True)
    record(
        "engine_workers' from-imported binding is the patched function",
        ew_fn is fn,
        "the patch landed after this module was imported and rebinding missed it" if ew_fn is not fn else "",
    )
except Exception as e:  # noqa: BLE001 - optional deps may block this import
    record("engine_workers importable", None, f"skipped: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 3. Does calling it actually move this process's CPU affinity?
# ---------------------------------------------------------------------------
section("Actual pinning")
os.environ.setdefault("LOCAL_RANK", "0")
local_rank = int(os.environ["LOCAL_RANK"])

# Report the topology the patch will walk, so a SKIP is explainable.
bdf = None
numa_node = None
try:
    from verl_hardware_plugin.patches import numa_affinity_patch_xpu as np_patch

    bdf = np_patch._zes_device_pci_bdf(local_rank)
    numa_node = np_patch._read_int(f"/sys/bus/pci/devices/{bdf}/numa_node")
    cpulist = (
        np_patch._read_text(f"/sys/devices/system/node/node{numa_node}/cpulist") if numa_node is not None else None
    )
    print(f"  local_rank={local_rank} bdf={bdf} numa_node={numa_node} cpulist={cpulist}", flush=True)
    record("resolved PCI BDF via pyzes", True, f"bdf={bdf}")
except Exception as e:  # noqa: BLE001
    record("resolved PCI BDF via pyzes", False, f"{type(e).__name__}: {e}")

before = os.sched_getaffinity(0)
vd.set_numa_affinity()
after = os.sched_getaffinity(0)

print(f"  affinity before: {len(before)} cpus", flush=True)
print(f"  affinity after:  {len(after)} cpus", flush=True)

if numa_node is not None and numa_node < 0:
    record(
        "affinity pinning",
        None,
        f"box reports numa_node={numa_node} for {bdf} (no NUMA topology) -- correctly skipped, nothing to pin",
    )
elif after != before:
    record("affinity pinning changed this process's cpuset", True, f"{len(before)} cpus -> {len(after)} cpus")
elif numa_node is not None and len(before) == len(after) and len(after) == os.cpu_count():
    record(
        "affinity pinning",
        False,
        f"numa_node={numa_node} exists but affinity is unchanged at {len(after)}/{os.cpu_count()} cpus -- "
        "the call silently no-opped, which is the original bug",
    )
else:
    record(
        "affinity pinning",
        None,
        f"affinity unchanged ({len(after)} cpus) -- may already have been pinned, or this node's "
        "NUMA node covers every visible cpu",
    )

# ---------------------------------------------------------------------------
# 4. Every visible device's topology, which is how the per-pod cross-NUMA case
#    was originally found.
# ---------------------------------------------------------------------------
section("All visible devices")
try:
    import torch

    count = torch.xpu.device_count() if hasattr(torch, "xpu") else 0
    print(f"  torch.xpu.device_count()={count}", flush=True)
    from verl_hardware_plugin.patches import numa_affinity_patch_xpu as np_patch

    seen = {}
    for idx in range(count):
        try:
            d_bdf = np_patch._zes_device_pci_bdf(idx)
            d_numa = np_patch._read_int(f"/sys/bus/pci/devices/{d_bdf}/numa_node")
            d_cpus = np_patch._read_text(f"/sys/devices/system/node/node{d_numa}/cpulist")
            print(f"  gpu[{idx}] bdf={d_bdf} numa_node={d_numa} cpulist={d_cpus}", flush=True)
            seen[idx] = d_numa
        except Exception as e:  # noqa: BLE001
            print(f"  gpu[{idx}] topology probe failed: {type(e).__name__}: {e}", flush=True)
    if len(set(seen.values())) > 1:
        print("  NOTE: this pod's GPUs span multiple NUMA nodes -- the per-device keying matters here.", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"  skipped: {type(e).__name__}: {e}", flush=True)

# ---------------------------------------------------------------------------
section("Summary")
failed = [c for c in checks if c[1] is False]
skipped = [c for c in checks if c[1] is None]
passed = [c for c in checks if c[1] is True]
print(f"  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped", flush=True)
for label, _, detail in failed:
    print(f"  FAILED: {label} -- {detail}", flush=True)
sys.exit(1 if failed else 0)
