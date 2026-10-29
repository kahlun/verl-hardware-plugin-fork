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
