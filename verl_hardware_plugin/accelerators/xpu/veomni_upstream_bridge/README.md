# VeOmni upstream bridge (temporary)

Everything in this folder exists only because VeOmni does not yet run on Intel XPU.
It is **meant to be deleted** once the matching VeOmni changes are released and
verl's pinned VeOmni includes them. Nothing here is XPU-engine logic: the engine
(`../engines/veomni_xpu.py`) stays after this folder is gone.

Each module monkeypatches **VeOmni** (never verl core) at runtime, so the plugin
works against an unmodified VeOmni checkout. Every module is a no-op once VeOmni
has the feature itself, so the folder can be removed at any point after upstream
lands, not only on a flag day.

| Module | What it bridges | Upstream change | Remove when |
|---|---|---|---|
| `device.py` | VeOmni reports `cpu` on XPU: no `IS_XPU_AVAILABLE`, `get_device_type()`, `get_dist_comm_backend()` (`xccl`), `stream_synchronize()`, and `helper.empty_cache()` skips XPU | [VeOmni#1387](https://github.com/ByteDance-Seed/VeOmni/pull/1387) (OPEN, review required) | verl's pinned VeOmni contains #1387 |
| `hub_fa2.py` | Ulysses SP on XPU: hub FA2 kernel major version, and `OP_REGISTRY` rows for `xpu` | VeOmni hub-FA2-on-XPU change (not yet submitted; branch `feat/xpu-hub-flash-attention`, patch `veomni_xpu_hub_fa2.patch`) | VeOmni has an `xpu` row in `_FA2_HUB_ROWS` and `_hub_flash_kernel_version` |
| `float8.py` (opt-in `VERL_XPU_VEOMNI_FLOAT8=tensorwise`) | torchao float8 training on XPU: converts Linear layers, makes VeOmni's fused SwiGLU MLP respect `Float8Linear`, pads the token dim to a multiple of 16; `VERL_XPU_VEOMNI_FLOAT8_ALLGATHER=1` adds the FSDP2 float8 all-gather | [VeOmni#1364](https://github.com/ByteDance-Seed/VeOmni/pull/1364) (draft, NVIDIA SM89+ only, rejects compile) | VeOmni float8 accepts XPU |
| `low_precision_rs.py` | VeOmni's BF16-transport / FP32-reduction ReduceScatter (`engine.low_precision_reduce_scatter_comm=True`) is device-agnostic but gated to CUDA; allows `xpu` | VeOmni#1167 added it with the CUDA gate; no XPU change upstream yet | VeOmni allows XPU there |
| `compile_layers.py` (opt-in `VERL_XPU_VEOMNI_COMPILE=1`) | `torch.compile` of each decoder layer in place; VeOmni's own compile option raises "CUDA-only" | none yet | VeOmni's decoder-block compile accepts XPU |
| `checkpoint_alias.py` | verl main imports `veomni.models.checkpoint_tensor_loading`; VeOmni moved it to `checkpoint.convert` | [VeOmni#1142](https://github.com/ByteDance-Seed/VeOmni/pull/1142). This one is a verl/VeOmni version skew, not XPU | verl updates the import |

## How to remove

1. Bump verl's VeOmni pin to a release containing the changes in the table.
2. Delete this folder.
3. Delete the `veomni_upstream_bridge.apply()` call at the top of
   `engines/veomni_xpu.py`.
4. Run the VeOmni SFT/GRPO smoke on XPU (see "Verifying" below).

## Verified results (stock VeOmni main `95f93e3`, 2026-11-12/14)

| Setup | Result |
|---|---|
| B60, 2 ranks: SFT with FSDP2 baseline, VeOmni SDPA, VeOmni hub-FA2 + Ulysses SP=2 | all rc=0; loss within ~0.002 of FSDP2 per step |
| B60, 2 ranks: GRPO with FSDP2, VeOmni SDPA, VeOmni SP=2 | all rc=0; ppo_kl ~1e-4, mismatch and response length match FSDP2 |
| CRI, 2 ranks: SFT bf16 eager and compiled decoder layers | rc=0, losses identical to ~1e-3 |
| CRI, 2 ranks: GRPO with VeOmni vs FSDP2 (Qwen2.5-0.5B, 3 steps) | both rc=0; mismatch 0.0039-0.0046 vs 0.0042-0.0047, step time 32/14/14 s vs 31/13.5/13.5 s. Reward was 0.0 in every step, so this shows the engine runs and tracks FSDP2, not that it learns |
| CRI, 2 ranks: GRPO with a moving reward (Qwen2.5-1.5B-Instruct, bs32 n8 resp512, lr 5e-6, 8 steps; last ~5 steps captured in the summary) | both rc=0. score/mean FSDP2 0.14 -> 0.51 -> 0.71 -> 0.71 -> 0.70, VeOmni 0.11 -> 0.56 -> 0.72 -> 0.68 -> 0.70; rollout-vs-trainer mismatch ~0.0032 both; grad_norm 0.55-0.96 both; step time 42-71 s (FSDP2) vs 48-76 s (VeOmni, noisier) |
| With the bridge disabled (`VERL_XPU_VEOMNI_BRIDGE=0`) | `Unknown backend: veomni` (engine cannot even register) |

Requirement on the CRI image (CRI_main_2675_ww44.2): VeOmni GRPO needs `transformers==5.17.0` and
`tokenizers>=0.23.1` (installed `--no-deps`); the image's 5.14.1 fails in `create_causal_mask`
(`allow_is_causal_skip`). The SFT path did not hit it.

## Low-precision status (CRI, 2 ranks, Qwen2.5-1.5B-Instruct, GSM8K; 2026-11-15)

| Feature | Result |
|---|---|
| float8 SFT, eager (`float8.py`) | rc=0; loss 0.586 -> 0.470 vs bf16 0.568 -> 0.467 (10 steps) |
| float8 SFT + FSDP2 float8 all-gather | rc=0; loss 0.586 -> 0.471; 392 Float8Linear calls before the first optimizer step |
| float8 SFT, compiled (`compile_layers.py`), gradient checkpointing off | rc=0; loss 0.584 -> 0.475; compiled bf16 0.567 -> 0.467. Compiled float8 with checkpointing on raises `CheckpointError` |
| low-precision ReduceScatter (`low_precision_rs.py`) | rc=0; "transport enabled for node-local shard group [0, 1]"; losses bit-identical to the default path (expected: gradients are bf16-valued and a 2-rank FP32 sum is exact). Speed not measured |
| float8 GRPO (VeOmni actor + ref), 8 steps | rc=0; reward 0.14 -> 0.83 (bf16 0.10 -> 0.72); rollout-vs-trainer mismatch 0.012 vs 0.0033 for bf16 (actor in float8, rollout in bf16); 71-92 s/step vs 63 s |
| float8 speed | slower than bf16 in every test (SFT ~26% compiled at ~9k tokens/card; GRPO ~30%). On B60 float8 matmul is 0.69x bf16 and a compiled float8 MLP 1.8x slower |

Note: verl's VeOmni engine config defaults `mixed_precision: true` (FP32 master weights, BF16 compute), so
the "bf16" baseline above is already mixed precision.

## Single-rank runs: not supported (any hardware)

verl's VeOmni engine hardcodes `data_parallel_mode="fsdp2"`, and VeOmni asserts `init_device="meta"` for FSDP2, but at
world size 1 FSDP is off and VeOmni's non-FSDP branch rejects `meta`. Verified on B60: `init_device=meta` fails with "Only FSDP
training supports init_device=meta"; `init_device=xpu` fails with "Please use model.accelerator.init_device: meta for FSDP2
training". Use 2 or more ranks. (A bridge for the non-FSDP list was tried and removed: it cannot help through verl.)

## Not bridged (VeOmni#1387 has these, verl's engine path does not use them)

`helper.py` profiler / full-determinism / `num_alloc_retries` branches (VeOmni's own
trainer only), `init_device` Literal widening (type hints), `train.sh`, docs, CI.
Running VeOmni's own trainer (not through verl) on XPU needs #1387 itself.

## Verifying the bridge against stock VeOmni

Run with `VERL_XPU_VEOMNI_BRIDGE=0` to skip the bridge. Against VeOmni without
#1387 that must fail (model stays on CPU); with #1387 it must pass. That is the
proof the bridge is what makes the difference, and that it is safe to delete.
