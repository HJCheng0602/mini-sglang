from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist

from ..base import BaseKVTransferBackend, KVTransferArgs, TransferStatus

class GlooTransferBackend(BaseKVTransferBackend):

    def __init__(self, backend_config:dict):
        self.backend_config = backend_config
        self.pending_transfers: Dict[int, TransferStatus] = {}

    def init_transfer(self, args:KVTransferArgs) -> None:
        self.pending_transfers[args.uid] = TransferStatus.BOOTSTRAPPING

        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")

    def send_kv_cache(self, args: KVTransferArgs, kv_data : List[torch.Tensor]) -> None:
        self.pending_transfers[args.uid] = TransferStatus.TRANSFERRING

        for data in kv_data:
            cpu_data = data.cpu()
            dist.send(tensor=cpu_data, dst=args.dst_rank)

        self.pending_transfers[args.uid] = TransferStatus.SUCCESS

    def recv_kv_cache(self, 
                        args: KVTransferArgs, 
                        kv_buffers: List[torch.Tensor]) -> None:
        self.pending_transfers[args.uid] = TransferStatus.TRANSFERRING
        for buffer in kv_buffers:
            cpu_buffer = torch.empty_like(buffer, device='cpu')
            dist.recv(tensor=cpu_buffer, src=args.src_rank)
            buffer.copy_(cpu_buffer)
        self.pending_transfers[args.uid] = TransferStatus.SUCCESS

    def poll(self, uid: int) -> TransferStatus:
        return self.pending_transfers.get(uid, TransferStatus.FAILED)
    
    def cleanup(self, uid):
        if uid in self.pending_transfers:
            del self.pending_transfers[uid]
    
    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
    

        
        
        