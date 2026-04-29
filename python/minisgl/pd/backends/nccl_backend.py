from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from ..base import BaseKVTransferBackend, KVTransferArgs, TransferStatus

class NCCLTransferBackend(BaseKVTransferBackend):

    def __init__(self, backend_config:dict):
        self.backend_config = backend_config
        self.transfer_group: Optional[dist.ProcessGroup] = None
        self.pending_transfers: Dict[int, TransferStatus] = {}
        self.transfer_stream = torch.cuda.Stream()

        # self._ensure_process_group()

    def init_transfer(self, args:KVTransferArgs) -> None:
        self.pending_transfers[args.uid] = TransferStatus.BOOTSTRAPPING

    def _ensure_process_group(self, world_size:int, rank:int, master_addr:str, master_port:int) -> None:
        if self.transfer_group is None:
            import os
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = str(master_port)

            dist.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank
            )
            self.transfer_group = dist.group.WORLD

    def send_kv_cache(
        self,
        args:KVTransferArgs,
        kv_data: List[torch.Tensor]
    ) -> None:
        self.pending_transfers[args.uid] = TransferStatus.TRANSFERRING

        with torch.cuda.stream(self.transfer_stream):
            ops = []
            for data in kv_data:
                contiguous_data = data.contiguous()
                ops.append(dist.isend(tensor=contiguous_data, dst=args.dst_rank, group=self.transfer_group))
            for op in ops:
                op.wait()
        self.pending_transfers[args.uid] = TransferStatus.SUCCESS

    def recv_kv_cache(self, args: KVTransferArgs, kv_buffers:List[torch.Tensor]) -> None:
        self.pending_transfers[args.uid] = TransferStatus.TRANSFERRING

        with torch.cuda.stream(self.transfer_stream):
            ops = []
            for buffer in kv_buffers:
                ops.append(dist.irecv(tensor=buffer, src=args.src_rank, group=self.transfer_group))
            for op in ops:
                op.wait()
        self.pending_transfers[args.uid] = TransferStatus.SUCCESS

    def poll(self, uid: int) -> TransferStatus:
        return self.pending_transfers.get(uid, TransferStatus.FAILED)
    
    def cleanup(self, uid: int) -> None:
        if uid in self.pending_transfers:
            del self.pending_transfers[uid]

    def shutdown(self):
        if self.transfer_group is not None:
            dist.destroy_process_group(self.transfer_group)
        