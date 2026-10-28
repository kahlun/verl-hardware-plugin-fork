# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Rank layout of the XPU XCCL checkpoint engine.

Pure Python: no accelerator and no process group. The module only imports on a
torch build that ships ``ProcessGroupXCCL``, so everything is skipped elsewhere.
Moving real weights over XCCL needs XPU hardware and a vLLM/SGLang rollout; that
is covered by the end-to-end run described in docs/accelerators/xpu/xccl_checkpoint_engine.md.
"""

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
