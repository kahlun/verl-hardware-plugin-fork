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

```

### Run with GPU access

```bash
# Find render group GID on host
RENDER_GID=$(getent group render | cut -d: -f3)
docker run --rm --device /dev/dri --group-add ${RENDER_GID} \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host --shm-size=16g \
  verl-intel-gpu:latest

```

## Example run case

setup dataset.
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k" >/tmp/gsm8k.log 2>&1


---
case 1

NUM_GPUS=2 MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct DATA_DIR=$HOME/data \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_fsdp2_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_fsdp2' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=2 


NUM_GPUS=2 MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct DATA_DIR=$HOME/data \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_fsdp2_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_fsdp2' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=2 \

---
case 2

NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
DATA_DIR=${DATA_DIR:-$HOME/data}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/gsm8k/train.parquet \
    data.val_files=$DATA_DIR/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_fsdp2_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_fsdp2' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} $@

## Dependencies

| Package | Version | Source |
|---------|---------|--------|
| PyTorch XPU | from vLLM xpu requirements | `https://download.pytorch.org/whl/xpu` |
| oneCCL runtime | installed in image via oneAPI / oneCCL bundle | bundled in image |
| vLLM | from Dockerfile `VLLM_VERSION` | prebuilt XPU wheel from `wheels.vllm.ai` |
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
