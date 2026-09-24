#!/bin/bash
# In-pod driver for scripts/verify_numa_patch_xpu.py, run via devctl.
#
# The prebuilt verl-intel-gpu images bake this plugin in at
# /opt/verl-hardware-plugin (installed `-e .`), so they carry whatever branch
# they were built from. This overlays the synced worktree on top of that, then
# installs pyzes, which the image does not ship: Dockerfile.intel_gpu installs
# level-zero's C library (level-zero / level-zero-devel .deb) but not the
# `pyzes` Python binding, so without this step the patch would correctly log
# "pyzes not available" and no-op.
#
# Usage (from the repo root, with an active devctl session on this worktree):
#   devctl test --gpu=2 --gpu-model=B60 --attach \
#     --image=<verl-intel-gpu image> \
#     --command="bash,/workspace/local/scripts/devctl_verify_numa.sh"

set -uo pipefail

echo "=== pod: $(hostname) ==="
echo "--- GPU devices visible to the container ---"
ls -la /dev/dri 2>&1 | head -20

echo
echo "--- overlay synced plugin worktree onto the image's editable install ---"
PLUGIN_DST=/opt/verl-hardware-plugin/verl_hardware_plugin
if [ -d "$PLUGIN_DST" ]; then
    cp -r /workspace/local/verl_hardware_plugin/. "$PLUGIN_DST"/
    echo "overlaid onto $PLUGIN_DST"
    ls -la "$PLUGIN_DST"/patches/
else
    echo "WARN: $PLUGIN_DST missing -- is this a verl-intel-gpu image?"
    ls -la /opt/ 2>&1 | head
fi

echo
echo "--- ensure pyzes >= 0.1.2 ---"
# The PCI family this patch depends on (zesDevicePciGetProperties,
# zes_pci_properties_t) landed in pyzes 0.1.2 via oneapi-src/level-zero#462.
# Confirmed on a B60 run: verl-intel-gpu:pr8-monkeypatch-20260924 ships 0.1.1,
# which has no PCI API at all, so pinning failed with a bare AttributeError.
if python3 -c "import pyzes; assert hasattr(pyzes, 'zesDevicePciGetProperties')" 2>/dev/null; then
    echo "pyzes already has the PCI API"
else
    echo "pyzes missing the PCI API -- installing >=0.1.2"
    uv pip install "pyzes>=0.1.2" 2>&1 | tail -3 || pip install "pyzes>=0.1.2" 2>&1 | tail -3
fi
python3 -c "
import pyzes
from importlib.metadata import version
print('pyzes', version('pyzes'), '->', pyzes.__file__)
print('has PCI API:', hasattr(pyzes, 'zesDevicePciGetProperties'))
" 2>&1 | tail -3

echo
echo "--- verification ---"
cd /opt/verl || cd /workspace/local
export VERL_PLATFORM=intel
export VERL_LOGGING_LEVEL=INFO
export LOCAL_RANK="${LOCAL_RANK:-0}"
python3 /workspace/local/scripts/verify_numa_patch_xpu.py
rc=$?

echo
echo "=== verify_numa_patch_xpu.py exit code: $rc ==="
exit $rc
