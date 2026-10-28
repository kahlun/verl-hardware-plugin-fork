# XCCL checkpoint engine (disaggregated trainer / rollout on Intel GPUs)

verl's `nccl` checkpoint engine cannot run on XPU: it imports cupy and hard-codes `cuda`. This plugin registers a
separate engine under the backend name `xccl` that moves weights over a torch `ProcessGroupXCCL`:

```
actor_rollout_ref.rollout.checkpoint_engine.backend=xccl
```

It needs no Ray collective backend, no cupy and no change to vLLM. Actor rank 0 hosts a `TCPStore`, publishes its
port and a per-round id through the usual `MasterMetadata`, and every rank builds its own `ProcessGroupXCCL` over it;
the default torch.distributed world is left alone. Bucketing, the ZMQ metadata channel and `send_weights` /
`receive_weights` follow the NCCL engine.

Code: `verl_hardware_plugin/accelerators/xpu/engines/xccl_checkpoint_engine.py`, registered from
`verl_hardware_plugin/accelerators/xpu/registration.py`.

## Settings that mattered

Verified once, with the engine living only in this plugin (verl core = upstream main, no XCCL engine in it), using
one-step-off GRPO (`verl.experimental.one_step_off_policy.main_ppo`), 2 trainer + 2 vLLM rollout GPUs on B60,
Qwen2.5-0.5B-Instruct, 3 steps in 333 s: four weight syncs (initial plus one per step; 1.85, 0.86, 0.73 and 0.73 s)
into live vLLM servers, no errors, `multi_sender` at its default, 32 CPUs, 52 GiB host memory.

| Setting | Why |
|---|---|
| At least 32 CPUs on the devctl job | With 16 the Ray rollout placement group (1 GPU + 2 CPU) never schedules (the task runner holds 10 CPUs, the trainer group reserves 6); the run looks hung. |
| `update_weights_bucket_megabytes` >= largest single weight | verl refuses a weight that does not fit (about 544 MB in fp32 for `embed_tokens` of Qwen2.5-0.5B). 1024 was used. |
| `multi_sender` left at its default `False` | With `True` and a real FSDP weight generator the sync stalled after the first bucket: the relay path waits on a host-blocking broadcast while the actor ranks also need each other for the next all-gather. |
| Each Ray worker on its own GPU | Provided by the plugin's `ray_device_pin_patch` (applied from `PlatformXPU.__init__`): the verified run logged `Pinned xpu:0` / `xpu:1` for the trainers and `xpu:2` / `xpu:3` for the checkpoint-engine workers, with `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR` unset. With the plugin bundled in the 2026-10-02 image (not inspected; its behaviour suggests it lacks this patch) over upstream-main verl core, a vLLM replica shared the trainer's card and the trainer died with `UR_RESULT_ERROR_OUT_OF_RESOURCES`; setting that variable was the workaround there. |

## Design notes

- **Subscription handshake.** Bucket metadata travels over ZeroMQ PUB/SUB next to the XCCL group. ZeroMQ drops a
  message published before the subscription has reached the publisher, and the XCCL barrier says nothing about that
  separate connection. Rank 0 therefore uses an XPUB socket and, after the barrier, waits until it has seen one
  subscription per consumer (`_wait_for_subscribers`) before it publishes. Consumers keep their SUB socket across syncs
  while the publisher's endpoint is unchanged. ZeroMQ reports a consumer leaving only when it was the last one on the
  topic, so the wait cannot see one of several consumers go away; consumers restarting mid-run is not supported.
- **Aligned bucket offsets.** The receiver rebuilds each weight with `chunk.view(dtype)`, which needs the chunk to start
  at a multiple of the element size, and a bucket can mix dtypes. Chunk starts are rounded up to 16 bytes by `_place`,
  which the sender and the relay both call, so their bucket boundaries stay identical.

## Swapping the transport later

Only four methods touch the transport: `_start_store`, `_create_process_group`, `_destroy_process_group` and the
module-level `_broadcast`. If Ray ships an XCCL backend in `ray.util.collective` and verl's own `nccl` engine becomes
device-agnostic, the simplest swap is to delete this engine and set `backend=nccl`. A Ray- or vLLM-backed variant would
replace those four methods and leave the bucketing and rank layout alone.

## Not covered

Multi-node (cross-node XCCL has known problems on oneCCL 2022.x), more than 4 GPUs, models above 0.5B, repeated runs.
