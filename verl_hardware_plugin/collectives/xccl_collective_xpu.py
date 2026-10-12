# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Collective-communication shim for PlatformXPU.get_collective_module().

verl.utils.rendezvous.ray_backend's create_nccl_communicator_in_ray() expects
a module shaped like cupy.cuda.nccl: a module-level get_unique_id() (called
once, by rank 0, before any communicator exists) and a
NcclCommunicator(ndev, commId, rank) class. NCCL's unique id is an opaque
byte blob with no listener behind it. XCCL has no such primitive --
torch's ProcessGroupXCCL is bootstrapped from a live TCPStore server
instead. So get_unique_id() here doubles as "start the TCPStore server",
and commId carries (host, port) rather than a byte blob.
"""

import torch
import torch.distributed as dist

_pending_store = None


def get_unique_id():
    global _pending_store
    _pending_store = dist.TCPStore(host_name="0.0.0.0", port=0, is_master=True, use_libuv=True)
    import ray

    host = ray.util.get_node_ip_address()
    return (host, _pending_store.port)


class NcclCommunicator:
    """Shape-compatible stand-in for cupy.cuda.nccl.NcclCommunicator, backed by XCCL."""

    def __init__(self, ndev, commId, rank):
        global _pending_store
        host, port = commId
        if rank == 0 and _pending_store is not None:
            store = _pending_store
        else:
            store = dist.TCPStore(host_name=host, port=port, is_master=False)
        _pending_store = None

        c10d = torch._C._distributed_c10d
        self._pg = c10d.ProcessGroupXCCL(store, rank, ndev)
        self._rank = rank

    def rank_id(self) -> int:
        return self._rank
