# Installation Guide

Intel XPU support ships as this external plugin package
(`verl_hardware_plugin`), not baked into the verl source tree — see
[`verl/plugin/platform/README.md`](https://github.com/verl-project/verl/blob/main/verl/plugin/platform/README.md)
in verl-core for the plugin pattern this follows.

## Prerequisites

- An Intel GPU with the Level Zero driver stack on the host (compute-runtime >= 26.18 and Level Zero).
  **A system oneAPI install is not required**: PyTorch's XPU wheel and vLLM's XPU wheel bring their own SYCL
  runtime and oneCCL (the `xccl` backend). This was checked by hiding `/opt/intel/oneapi` and the oneAPI
  environment variables: platform detection, an `xccl` all_reduce and a GRPO training step all still worked.
- PyTorch with XPU support (`torch.xpu.is_available() == True`).
- vLLM with XPU kernels, from vLLM's XPU wheel index (`pip install "vllm==<version>+xpu"
  --extra-index-url https://wheels.vllm.ai/<version>/xpu`) or built from source with
  `VLLM_TARGET_DEVICE=xpu`. The default PyPI `pip install vllm` wheel does not ship XPU kernels.

**verl-core version note:** the one `PlatformBase` hook this plugin
implements, `profiler_markers()`, was merged into `verl-project/verl:main` on
2026-10-08. A checkout from before that date still installs and runs this
plugin, but the hook silently no-ops on it: ITT ranges are not emitted, and
the `tool_config` allowlist in `engine_workers.py` that the same change
lifted drops a `tool_config.vtune` entry. `VtuneProfiler` itself is
registered by a plugin-side monkeypatch and works either way. Attention
padding needs nothing from this plugin: verl core uses
its own pure-PyTorch `attention_padding_utils` whenever `flash_attn` is
unavailable. The `xccl` reduce_avg workaround is likewise unaffected — it's a
plugin-side monkeypatch over `torch.distributed.all_reduce`
(`accelerators/xpu/patches/reduce_avg_allreduce_patch.py`), not a
`PlatformBase` hook.

## 1. Install verl and verl-hardware-plugin

Pick one. Both give the same environment (checked: torch 2.13.0+xpu, vLLM 0.29.0+xpu, transformers 5.17.0).

### Option A: Docker (recommended)

The Intel GPU image is maintained with verl-core, in
[`docker/xpu/`](https://github.com/verl-project/verl/tree/main/docker/xpu). It installs verl and this plugin
together, so there is nothing else to do. Build and run steps are in that directory's README.

### Option B: From source

verl pins its own versions, and `pip install -e .` on a verl checkout can replace the XPU build of torch with a
different one (it also downgrades `transformers` and `tensordict` to verl's pins). Install the XPU dependency
set the way verl-core documents it, in
[`docker/xpu/README.md` ("Installing Without Docker")](https://github.com/verl-project/verl/blob/main/docker/xpu/README.md),
which uses verl's `requirements-xpu.txt`. Then install this plugin on top of that environment:

```bash
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install --no-deps -e .
```

`--no-deps` keeps the plugin install from touching the torch/verl already in place. The plugin needs no
environment variable: it is auto-discovered by verl through the `verl.plugins` setuptools entry_points
group declared in this repo's `pyproject.toml`
(`[project.entry-points."verl.plugins"] hardware = "verl_hardware_plugin"`), which verl loads by default
(`VERL_USE_EXTERNAL_PLUGINS=auto`).

### Platform selection

With the plugin installed, verl picks the Intel platform by itself on a host where `torch.xpu` is available.
`VERL_PLATFORM=intel` is only an override, for forcing the choice (for example on a host with several kinds of
accelerator). `VERL_USE_EXTERNAL_PLUGINS=none` disables discovery, e.g. to isolate a bug to this plugin; verl then
falls back to `nvidia`.

## 2. Verify the Install

```bash
python3 -c "
from verl.plugin.platform import get_platform
p = get_platform()
print('device:', p.device_name, '/ vendor:', p.vendor_name)
"
```

Expected on Intel GPU: `device: xpu / vendor: intel`. If it instead falls
back to `nvidia`, either the plugin was not discovered (check that `pip install`
completed without error and that no `VERL_USE_EXTERNAL_PLUGINS=none` is set) or
torch lost its XPU build (check `python3 -c "import torch; print(torch.__version__, torch.xpu.is_available())"`;
a plain version like `2.14.1` without `+xpu` means a dependency install replaced it).

## Next Steps

Follow the [Quick Start](./quick_start.md).
