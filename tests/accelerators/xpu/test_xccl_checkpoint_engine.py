# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Rank layout of the XPU XCCL checkpoint engine.

Pure Python: no accelerator and no process group. The module only imports on a
torch build that ships ``ProcessGroupXCCL``, so everything is skipped elsewhere.
Moving real weights over XCCL needs XPU hardware and a vLLM/SGLang rollout; that
is covered by the end-to-end run described in docs/accelerators/xpu/xccl_checkpoint_engine.md.
"""

import time

import pytest

engine = pytest.importorskip(
    "verl_hardware_plugin.accelerators.xpu.engines.xccl_checkpoint_engine",
    reason="needs a torch build with XPU and ProcessGroupXCCL",
)

MASTER = engine.MasterMetadata(zmq_ip="10.0.0.1", zmq_port=5555, store_port=6666, group_id="g", multi_sender=False)


def _meta(*node_ids, multi_sender=False):
    master = engine.MasterMetadata(
        zmq_ip=MASTER.zmq_ip,
        zmq_port=MASTER.zmq_port,
        store_port=MASTER.store_port,
        group_id=MASTER.group_id,
        multi_sender=multi_sender,
    )
    return [engine.WorkerMetadata(node_id=n, master=master if i == 0 else None) for i, n in enumerate(node_ids)]


def test_registered_under_xccl():
    from verl.checkpoint_engine import CheckpointEngineRegistry

    assert CheckpointEngineRegistry.get("xccl") is engine.XCCLCheckpointEngine


def test_multi_sender_is_off_by_default():
    import inspect

    default = inspect.signature(engine.XCCLCheckpointEngine.__init__).parameters["multi_sender"].default
    assert default is False


def test_single_sender_only_actor_rank0_joins():
    actor_kwargs, rollout_kwargs = engine.XCCLCheckpointEngine.build_topology(
        actor_wg_world_size=2, rollout_world_size=2, metadata=_meta("a", "a")
    )
    assert actor_kwargs["rank"] == [0, -1]
    assert rollout_kwargs["rank"] == [1, 2]
    assert actor_kwargs["world_size"] == [3, 3]
    assert rollout_kwargs["num_senders"] == [1, 1]


def test_multi_sender_admits_same_node_actors_only():
    actor_kwargs, rollout_kwargs = engine.XCCLCheckpointEngine.build_topology(
        actor_wg_world_size=3, rollout_world_size=2, metadata=_meta("a", "a", "b", multi_sender=True)
    )
    # actor 1 shares rank 0's node and relays; actor 2 is on another node and sits out
    assert actor_kwargs["rank"] == [0, 1, -1]
    assert rollout_kwargs["rank"] == [2, 3]
    assert actor_kwargs["num_senders"] == [2, 2, 2]


def test_multi_sender_degrades_to_single_when_no_local_peers():
    actor_kwargs, rollout_kwargs = engine.XCCLCheckpointEngine.build_topology(
        actor_wg_world_size=2, rollout_world_size=1, metadata=_meta("a", "b", multi_sender=True)
    )
    assert actor_kwargs["rank"] == [0, -1]
    assert rollout_kwargs["rank"] == [1]


# --- bucket layout: offsets must stay aligned for mixed dtypes --------------------------------------------------


def test_place_aligns_chunk_starts_and_flushes_when_full():
    place, align = engine._place, engine._ALIGN
    assert place(0, 6, 64) == (0, False)
    # 6 bytes in, the next chunk starts at the next aligned offset
    assert place(6, 4, 64) == (align, False)
    assert place(align, 4, 64) == (align, False)
    # no room left after alignment: flush and start a new bucket at 0
    assert place(60, 8, 64) == (0, True)
    assert place(50, 16, 64) == (0, True)
    # a chunk that exactly fills an empty bucket fits
    assert place(0, 64, 64) == (0, False)


def test_unaligned_slice_cannot_be_viewed_as_a_wider_dtype():
    # The premise of the alignment rule: this is what verl's merge_weight_chunks does to every received chunk.
    import torch

    buf = torch.zeros(16, dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        buf[2:6].view(torch.float32)
    assert buf[4:8].view(torch.float32).shape == (1,)


def test_mixed_dtype_bucket_round_trips_through_merge_weight_chunks():
    import asyncio

    import torch

    from verl.checkpoint_engine.base import TensorMeta, merge_weight_chunks

    # A 3-element bf16 weight (6 bytes) followed by fp32 weights: without padding the first fp32 would start at
    # byte 6, which cannot be viewed as float32.
    weights = {
        "a": torch.arange(3, dtype=torch.bfloat16),
        "b": torch.tensor([1.5, 2.5], dtype=torch.float32),
        "c": torch.arange(5, dtype=torch.bfloat16),
        "d": torch.tensor([7.0], dtype=torch.float32),
    }
    bucket_size = 256
    bucket = torch.zeros(bucket_size, dtype=torch.uint8)
    metas, offset = [], 0
    for name, weight in weights.items():
        raw = weight.view(-1).view(torch.uint8)
        start, flush = engine._place(offset, raw.numel(), bucket_size)
        assert not flush
        offset = start
        bucket[offset : offset + raw.numel()] = raw
        metas.append(
            TensorMeta(
                name=name, shape=weight.shape, dtype=weight.dtype, chunk_offset=0, chunk_size=raw.numel(), offset=offset
            )
        )
        offset += raw.numel()
        assert start % weight.element_size() == 0

    async def chunks():
        for meta in metas:
            yield meta, bucket[meta.offset : meta.offset + meta.chunk_size]

    async def collect():
        return {name: weight async for name, weight in merge_weight_chunks(chunks(), bucket_size)}

    rebuilt = asyncio.run(collect())
    assert set(rebuilt) == set(weights)
    for name, weight in weights.items():
        assert torch.equal(rebuilt[name], weight), name


# --- ZeroMQ: the publisher must know every consumer has subscribed before it publishes --------------------------


def _xpub_and_subs(n_subs):
    import zmq

    ctx = zmq.Context()
    pub = ctx.socket(zmq.XPUB)
    pub.setsockopt(zmq.XPUB_VERBOSE, 1)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    subs = []
    for _ in range(n_subs):
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVTIMEO, 5000)
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "bucket_metadata")
        subs.append(sub)
    return ctx, pub, subs


def test_wait_for_subscribers_counts_every_consumer_and_nothing_is_dropped():
    import zmq

    ctx, pub, subs = _xpub_and_subs(3)
    try:
        assert engine._wait_for_subscribers(pub, 0, 3, timeout_s=10) == 3
        # published right after the wait returns: every consumer must get it
        pub.send_string("bucket_metadata", flags=zmq.SNDMORE)
        pub.send_pyobj({"length": 7})
        for sub in subs:
            assert sub.recv_string() == "bucket_metadata"
            assert sub.recv_pyobj() == {"length": 7}
    finally:
        for sock in [pub, *subs]:
            sock.close(linger=0)
        ctx.term()


def test_wait_for_subscribers_times_out_instead_of_publishing_into_the_void():
    ctx, pub, subs = _xpub_and_subs(1)
    try:
        with pytest.raises(TimeoutError, match="1 of 2"):
            engine._wait_for_subscribers(pub, 0, 2, timeout_s=0.5)
    finally:
        for sock in [pub, *subs]:
            sock.close(linger=0)
        ctx.term()


def test_wait_for_subscribers_applies_an_unsubscribe_before_deciding():
    import zmq

    ctx, pub, subs = _xpub_and_subs(1)
    try:
        assert engine._wait_for_subscribers(pub, 0, 1, timeout_s=10) == 1
        subs[0].setsockopt_string(zmq.UNSUBSCRIBE, "bucket_metadata")
        time.sleep(0.5)  # let the unsubscribe frame reach the publisher
        # the only consumer unsubscribed: the count drops to 0, so expecting 1 again must time out
        with pytest.raises(TimeoutError):
            engine._wait_for_subscribers(pub, 1, 1, timeout_s=1.0)
    finally:
        for sock in [pub, *subs]:
            sock.close(linger=0)
        ctx.term()


def test_consumer_keeps_its_subscription_while_the_endpoint_is_unchanged():
    consumer = object.__new__(engine.XCCLCheckpointEngine)
    consumer.is_master = False
    consumer.topic = "bucket_metadata"
    consumer._zmq_endpoint = None
    meta = engine.MasterMetadata(zmq_ip="127.0.0.1", zmq_port=59999, store_port=0, group_id="g", multi_sender=False)

    consumer._connect_zmq_client(meta)
    first = consumer.socket
    consumer._connect_zmq_client(meta)
    assert consumer.socket is first, "a new SUB socket per sync leaks one and loses the established subscription"

    moved = engine.MasterMetadata(zmq_ip="127.0.0.1", zmq_port=59998, store_port=0, group_id="g", multi_sender=False)
    consumer._connect_zmq_client(moved)
    assert consumer.socket is not first
    assert first.closed
    consumer.socket.close(linger=0)
