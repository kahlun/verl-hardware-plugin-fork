# Installation Guide

Intel XPU support ships as this external plugin package
(`verl_hardware_plugin`), not baked into the verl source tree — see
[`verl/plugin/platform/README.md`](https://github.com/verl-project/verl/blob/main/verl/plugin/platform/README.md)
in verl-core for the plugin pattern this follows.

## Bare Metal / Existing Environment

```bash
pip install -e /path/to/verl-hardware-plugin
```

That's it — no environment variable is required. The plugin is
auto-discovered by verl through the `verl.plugins` setuptools entry_points
group declared in this repo's `pyproject.toml`
(`[project.entry-points."verl.plugins"] hardware = "verl_hardware_plugin"`),
which verl loads by default (`VERL_USE_EXTERNAL_PLUGINS=auto`). Set
`VERL_USE_EXTERNAL_PLUGINS=none` to disable discovery, e.g. to isolate a bug
to this plugin, or `VERL_PLATFORM=intel` to force platform selection instead
of relying on auto-detection.

Prerequisites this plugin does not install for you:

- PyTorch with XPU support (`torch.xpu.is_available() == True`)
- vLLM built from source with `VLLM_TARGET_DEVICE=xpu` (vLLM's `pip` wheels
  do not ship XPU kernels)
- oneCCL runtime for the `xccl` distributed backend

## Docker

A prebuilt image definition is at
[`docker/intel_gpu/`](../../docker/intel_gpu/) in this repo — it clones
verl-core at a pinned ref, builds vLLM from source for XPU, and installs this
plugin on top. See [`docker/intel_gpu/README.md`](../../docker/intel_gpu/README.md)
for build/run instructions and the full software stack table.

## Verifying the Install

```bash
python3 -c "
from verl.plugin.platform import get_platform
p = get_platform()
print('device:', p.device_name, '/ vendor:', p.vendor_name)
"
```

Expected on Intel GPU: `device: xpu / vendor: intel`. If it instead falls
back to `nvidia`, the plugin was not discovered — check that `pip install`
completed without error and that no `VERL_USE_EXTERNAL_PLUGINS=none` is set
in the environment.
