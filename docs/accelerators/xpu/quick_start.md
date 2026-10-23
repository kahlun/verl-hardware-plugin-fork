# Quick Start

Training example for running verl on Intel GPU (Arc / Arc Pro) with this
plugin installed (see [`install_guidance.md`](install_guidance.md) first).

Current software and hardware scope:

- Runtime mode: **Colocate** (FSDP actor + vLLM rollout on the same device).
- Inference engine: **vLLM** validated. SGLang weight sync — Work in
  Progress on Intel GPU.
- Trainer backend: **FSDP**, **FSDP2**. Megatron — Work in Progress.
- Algorithms: GRPO, PPO, SFT — validated on GSM8K / Qwen2.5.
- Hardware targets:
  - Intel Arc Pro B60 (Battlemage, 24 GB) — validated, 1-GPU and 2-GPU
  - Intel Arc Pro B70 (2x GPU, 32 GB each)
  - Multi-node — Work in Progress

## 1. Verify Platform Resolution

```bash
python3 -c '
from verl.plugin.platform import get_platform
p = get_platform()
print("device:  ", p.device_name)
print("vendor:  ", p.vendor_name)
print("backend: ", p.communication_backend_name())
print("ray res: ", p.ray_resource_name())
print("ipc:     ", p.is_ipc_supported())
'
```

Expected output:

```text
device:   xpu
vendor:   intel
backend:  xccl
ray res:  GPU
ipc:      False
```

## 2. Prepare Data

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

## 3. Launch Training

```bash
export ZE_AFFINITY_MASK=0              # select physical device indices; for two GPUs use 0,1 together with
                                       # trainer.n_gpus_per_node=2 and +ray_kwargs.ray_init.num_gpus=2
unset ONEAPI_DEVICE_SELECTOR           # must NOT be set

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    +ray_kwargs.ray_init.num_gpus=1
```

This is a smoke test: `total_training_steps=2` stops after two steps (without it, `total_epochs=1` over GSM8K at
batch 16 is about 467 steps). Raise it for a longer run. A `DataLoader worker ... is killed by signal` traceback
printed at exit is shutdown noise after training has finished.

Expected (Qwen2.5-0.5B-Instruct, GSM8K, batch 16, 1x Arc Pro B60). The first step includes warm-up; the second shows
steady state:

```text
step 1:  timing_s/step: ~16-21   perf/throughput: ~190-235 tok/s
step 2:  timing_s/step: ~8-8.5   perf/throughput: ~440-465 tok/s
timing_per_token_ms/gen: ~1.0    perf/max_memory_allocated_gb: ~9.2
```

(Three runs on 2026-10-21 to 2026-10-23: vLLM 0.29.0+xpu, torch 2.13.0+xpu, this plugin at `e5ab532`, verl `main` at
different commits. Timings vary with the stack and the host.)

Two GPUs: set `ZE_AFFINITY_MASK=0,1`, `trainer.n_gpus_per_node=2` and `+ray_kwargs.ray_init.num_gpus=2`, and keep
everything else the same. At this small batch the step time does not drop, because fixed per-step overhead dominates.
`perf/throughput` is reported per GPU, so it is about half of the 1-GPU value; use a larger batch to see scaling.

```text
step 1:  timing_s/step: ~16   perf/throughput: ~125 tok/s per GPU
step 2:  timing_s/step: ~8    perf/throughput: ~235 tok/s per GPU
perf/max_memory_allocated_gb: ~5.4 per GPU
```

## 4. Run the Plugin Test Suite

```bash
pytest tests/test_plugin_registration.py -k xpu -v
```

Expected: all XPU cases pass. These run on any host, with or without an
Intel GPU attached.

## Feature Support Matrix

| Category | Status | Notes |
|---|---|---|
| Runtime mode | Colocate | FSDP/FSDP2 actor + vLLM rollout on same GPU(s) |
| Inference engine | vLLM validated | SGLang: `update_weights` — Work in Progress |
| Trainer backend | FSDP, FSDP2 | Megatron — Work in Progress |
| Algorithms | GRPO, PPO, SFT | Validated on GSM8K / Qwen2.5 |
| Hardware (1-GPU) | Validated | Arc Pro B60: ~8 s/step steady state, ~450 tok/s (batch 16, Qwen2.5-0.5B, vLLM 0.29.0) |
| Hardware (2-GPU) | Validated | Arc Pro B60 x2: ~8 s/step steady state at the same batch (~235 tok/s per GPU); FSDP2 and sleep mode also run |
| Multi-node | Work in Progress | Two-node collectives work only with `CCL_ENABLE_SYCL_KERNELS=0` (then `ReduceOp.AVG` is unavailable); with oneCCL defaults a 64 MiB `all_reduce` across nodes hangs. A multi-node verl training run is not validated |
| Profiler | Intel VTune (ITT) — Validated | `profiler_markers()` hook merged into verl-core `main` on 2026-10-08; VTune collector capture itself still untested, see [profiling.md](./profiling.md) |

## Next Steps

See [Profiling](./profiling.md) to capture an Intel VTune (ITT) trace.
