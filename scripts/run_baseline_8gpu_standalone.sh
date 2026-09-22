#!/usr/bin/env bash
# Standalone wrapper around scripts/baseline_grpo_gsm8k.sh for running the
# Acceptance Baseline on a machine that isn't going through devctl (i.e. no
# session-PVC mount, no cluster registry credentials assumed).
#
# Run this INSIDE the container (see the `docker run` invocation this ships
# alongside). Everything it needs (verl-core, this plugin, the baseline
# script itself) is already baked into the image at /opt/verl and
# /opt/verl-hardware-plugin -- this script only does data prep + env setup,
# then calls the real baseline script unmodified.
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== platform check ==="
python3 -c "
from verl.plugin.platform import get_platform
p = get_platform()
print('PLATFORM:', p.device_name, p.vendor_name)
"

echo "=== vLLM version ==="
python3 -c "import vllm; print('vllm:', vllm.__version__)"

# baseline_grpo_gsm8k.sh hardcodes CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --
# that's a CUDA-only env var and does nothing on XPU (the real one is
# ZE_AFFINITY_MASK). Harmless no-op here since we want all 8 GPUs visible
# anyway, but don't rely on it to *restrict* devices on this platform.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3,4,5,6,7}
unset ONEAPI_DEVICE_SELECTOR || true

# Level-Zero IPC workarounds -- needed at 2 GPUs already, more load-bearing
# at 8 (see docker/intel_gpu/README.md "Known Workarounds").
export CCL_ATL_SHM=1
export CCL_BUFFER_CACHE=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_TOPO_ALGO=0

# Level-Zero VA-space pressure scales with colocated Ray worker count --
# 8 GPUs means 8x the workers of what we validated at 2 GPUs.
export RAY_memory_monitor_refresh_ms=0
export RAY_NUM_PRESTART_PYTHON_WORKERS=0

echo "=== data prep (GSM8K) ==="
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}
mkdir -p "$DATA_DIR"
cd /opt/verl
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"

# MODEL_DIR: baseline_grpo_gsm8k.sh defaults to a local path
# (/workspace/Qwen3-0.6B) that won't exist here. Point it at the HF hub id
# directly instead -- verl resolves actor_rollout_ref.model.path via
# from_pretrained, which accepts a hub id exactly like a local path.
export MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3-0.6B}
export DATA_DIR="$DATA_DIR"
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# trainer.logger='["console","swanlab"]' is hardcoded in the baseline
# script -- needs SWANLAB_API_KEY set (or `swanlab login` already run in
# this container/its mounted $HOME) or the run will fail/hang trying to
# authenticate. Not something this wrapper can do for you.
if [ -z "${SWANLAB_API_KEY:-}" ] && [ ! -f "$HOME/.swanlab/config" ]; then
  echo "WARNING: no SWANLAB_API_KEY and no ~/.swanlab/config found." >&2
  echo "  scripts/baseline_grpo_gsm8k.sh hardcodes trainer.logger=[console,swanlab]." >&2
  echo "  Set SWANLAB_API_KEY, or this run will likely fail at logger init." >&2
fi

echo "=== Acceptance Baseline: GRPO/GSM8K/Qwen3-0.6B, NGPUS_PER_NODE=${NGPUS_PER_NODE} ==="
cd /opt/verl-hardware-plugin
# vLLM >=0.29.0 (this image's pin) added AutoWeightsLoader._check_skipped_aliases():
# each load_weights() call now validates tied-weight completeness on its own.
# verl's bucketed weight-sync calls load_weights() once per bucket with no
# accumulation (verl/workers/rollout/vllm_rollout/utils.py), so if
# lm_head.weight and its tied target model.embed_tokens.weight land in
# different buckets, this crashes with "was skipped because it is tied to
# ... but ... was not found in the checkpoint". Doesn't exist in vLLM 0.27.0.
# Not XPU-specific -- would hit any device with a tied-embedding model.
# Workaround: make the bucket bigger than the whole (small) model so
# everything ships in one call. Qwen3-0.6B easily fits under 8192 MB.
bash scripts/baseline_grpo_gsm8k.sh actor_rollout_ref.rollout.update_weights_bucket_megabytes=8192
