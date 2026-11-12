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
| `checkpoint_alias.py` | verl main imports `veomni.models.checkpoint_tensor_loading`; VeOmni moved it to `checkpoint.convert` | [VeOmni#1142](https://github.com/ByteDance-Seed/VeOmni/pull/1142). This one is a verl/VeOmni version skew, not XPU | verl updates the import |

## How to remove

1. Bump verl's VeOmni pin to a release containing the changes in the table.
2. Delete this folder.
3. Delete the `veomni_upstream_bridge.apply()` call at the top of
   `engines/veomni_xpu.py`.
4. Run the VeOmni SFT/GRPO smoke on XPU (see "Verifying" below).

## Not bridged (VeOmni#1387 has these, verl's engine path does not use them)

`helper.py` profiler / full-determinism / `num_alloc_retries` branches (VeOmni's own
trainer only), `init_device` Literal widening (type hints), `train.sh`, docs, CI.
Single-process runs (world size 1, no FSDP) are NOT bridged: stock VeOmni's `build_parallelize_model`
rejects any `init_device` outside `cuda/npu/mlu` when FSDP is off (an inline list that #1387 extends with
`xpu`). Verified on CRI 2026-11-12: world size 1 fails with "Only FSDP training supports `init_device=meta`".
Use >= 2 ranks until #1387 lands.

Running VeOmni's own trainer (not through verl) on XPU needs #1387 itself.

## Verifying the bridge against stock VeOmni

Run with `VERL_XPU_VEOMNI_BRIDGE=0` to skip the bridge. Against VeOmni without
#1387 that must fail (model stays on CPU); with #1387 it must pass. That is the
proof the bridge is what makes the difference, and that it is safe to delete.
