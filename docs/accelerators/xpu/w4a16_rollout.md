# INT4 (W4A16) rollout on Intel XPU

Last updated: 10/23/2026.

verl's low-precision rollout (`actor_rollout_ref.rollout.quantization`) has the trainer quantize its
weights at every refit, using compressed-tensors, and the rollout engine load them. This plugin adds one
preset to that list on Intel XPU:

```bash
actor_rollout_ref.rollout.quantization.linear=W4A16
```

`W4A16` is weight-only INT4: symmetric, group size 128, packed 8 codes per int32 (compressed-tensors
`pack-quantized`). Training stays in BF16. On XPU, vLLM runs it through `XPUwNa16LinearKernel`
(`torch.ops._xpu_C.int4_gemm_w4a16`).

## Requirements

- A verl whose FSDP engine implements `set_rollout_weight_quantizer`. Before that, only the Megatron
  engine did, and an FSDP run stops at startup with
  `NotImplementedError: ... does not support trainer-side rollout quantization`.
- vLLM with the XPU build (tested: 0.29.0) and `compressed-tensors` (tested: 0.17.0).

## How it is registered

`patches/w4a16_rollout_preset_patch.py` adds `"W4A16"` (no QAT mode) to
`verl.utils.qat.compressed_tensors.SUPPORTED_SCHEMES`, from `PlatformXPU.__init__` like the other XPU
patches. Everything else is existing verl code: the config builder, the trainer-side exporter and the
engine refit path.

## What to expect

INT4 round-to-nearest is lossy, and the rollout drifts from the BF16 trainer much more than with FP8.
Measured on one Arc Pro B60, as the mean `|p_rollout - p_trainer|` of sampled tokens
(`training/rollout_probs_diff_mean` in a run):

| model | BF16 rollout | W4A16 (verl exporter, round-to-nearest) |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.004 | 0.076 to 0.083 (10-step GRPO) |
| Qwen2.5-1.5B-Instruct (2 GPUs) | 0.003 to 0.004 | 0.048 at step 1, then 0.08 to 0.17 (8-step GRPO) |

In one 8-step 1.5B run (256-token responses) W4A16 became unstable without rollout correction: the mean
response length fell from about 210 to about 22 tokens at steps 2 and 3 (partly recovering to about 140 by
step 8) and the gradient norm rose from about 1 to 4-22; the BF16 run on the same setup stayed flat. Later runs
on a newer image (768 to 1024-token responses, 3 to 6 steps, mismatch 0.05 to 0.07) did not collapse: response
length stayed at 176 to 409 tokens, and with token-level importance sampling (threshold 2) the gradient norm
stayed near 1. These are single short runs, so they show neither that W4A16 is stable nor that it is unstable.

On that newer image (`verl-xpu-neat`, vLLM 0.29, torch 2.13) W4A16 generation was not faster than BF16 at
1.5B (8 to 18 s per step for both), and the weight sync took 3.5 to 4.2 s against 4.0 to 4.8 s.

## Limits

- FSDP trainer, dense and non-LoRA weights, rollout TP=1. Not tested: MoE, LoRA, TP>1, Megatron.
- No QAT mode. Combining it with QAT should be rejected by verl's QAT-mode check at startup (not tested).
- Tested with vLLM only.
