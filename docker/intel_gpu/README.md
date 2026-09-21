# verl-hardware-plugin on Intel GPU

## Supported Hardware

- Intel Arc Pro B-Series (Battlemage)

## What This Image Is

This builds verl core (cloned at a pinned ref) with this plugin
(`verl_hardware_plugin`) installed on top of it, from this repo as the build
context. Unlike the CUDA/ROCm/Ascend images that live in `verl-core`'s own
`docker/` directory, this one is intentionally on the other side of the
plugin boundary: it starts from the plugin and pulls verl in, not the other
way round.

The plugin is auto-discovered by verl via the `verl.plugins` setuptools
entry_points group declared in this repo's `pyproject.toml` — no
`VERL_USE_EXTERNAL_MODULES` env var needed. See `verl/plugin/platform/README.md`
in verl-core for the discovery mechanism.

## Quick Start

### Build the Docker image

```bash
# Standard build
docker build -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .

# Behind a corporate proxy
docker build \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .

# Pin a different verl ref (e.g. once verl-project/verl#7917 merges upstream)
docker build \
  --build-arg VERL_GIT_REPO=https://github.com/verl-project/verl.git \
  --build-arg VERL_REF=v0.8.0 \
  -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .
```

### Run with GPU access

```bash
# Find render group GID on host
RENDER_GID=$(getent group render | cut -d: -f3)

docker run -it --rm \
  --device /dev/dri --group-add ${RENDER_GID} \
  --shm-size 16g \
  -v $HOME/data:/root/data \
  verl-intel-gpu:latest
```

### Runtime env override

For machine-specific or temporary settings (proxy, debug flags), inject them
explicitly at `docker run` time with `-e`/`--env-file` instead of baking values
into the image:

```bash
docker run -it --rm --device /dev/dri --group-add ${RENDER_GID} \
  --shm-size 16g -v $HOME/data:/root/data \
  -e CCL_ATL_SHM=1 \
  verl-intel-gpu:latest
```

## Dependencies

| Package | Version | Source |
|---------|---------|--------|
| PyTorch XPU | from vLLM xpu requirements | `https://download.pytorch.org/whl/xpu` |
| oneCCL runtime | installed in image via oneAPI / oneCCL bundle | bundled in image |
| vLLM | from Dockerfile `VLLM_VERSION` | built from source with `VLLM_TARGET_DEVICE=xpu` |
| verl core | from Dockerfile `VERL_GIT_REPO`/`VERL_REF` | git-cloned, `pip install --no-deps -e .` |
| verl deps | from `requirements-intel-gpu.txt` | PyPI and extra indexes |
| verl-hardware-plugin (this repo) | local build context | `pip install --no-deps -e .` |

Runtime sanity checks validated on this image:

- `torch.xpu.is_available() == True`
- `from vllm.platforms import current_platform` reports `xpu`
- oneCCL runtime is available via `CCL_ROOT` and `libccl.so.1`
- `python3 -c "from verl.plugin.platform import get_platform; print(get_platform().device_name)"` reports `xpu`

## Backend Policy

- Default rollout backend on Intel GPU is vLLM.
- sglang is not the default path for Intel GPU in this image.
- Separate image/profile will be released when SGLang is validated on Intel GPU.

## Known Workarounds (pre-DLE 2026.0 driver)

Multi-GPU requires these environment variables due to Level Zero IPC limitations:

```bash
export CCL_ATL_SHM=1        # Route collectives via /dev/shm
export CCL_BUFFER_CACHE=0    # Prevent stale IPC handle cache
```

Also commonly required in multi-GPU runs:

```bash
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0
```

These should be treated as temporary runtime workarounds and revisited when upgrading
to newer driver and PyTorch releases.
