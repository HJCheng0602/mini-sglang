from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..base import BaseKVTransferBackend, KVTransferArgs, TransferStatus


class NCCLTransferBackend(BaseKVTransferBackend):

    def __init__(self, backend_config: dict):
        self.backend_config = backend_config
        self.transfer_group: Optional[dist.ProcessGroup] = None
        self.pending_transfers: Dict[int, Tuple[TransferStatus, torch.cuda.Event, List[torch.Tensor]]] = {}
        self.transfer_stream = torch.cuda.Stream()

    def init_transfer(self, args: KVTransferArgs) -> None:
        self.pending_transfers[args.uid] = (TransferStatus.BOOTSTRAPPING, torch.cuda.Event(), [])
        self._ensure_process_group()

    def _ensure_process_group(self) -> None:
        if self.transfer_group is None:
            if dist.is_initialized():
                self.transfer_group = dist.group.WORLD
            else:
                dist.init_process_group(backend="nccl")
                self.transfer_group = dist.group.WORLD

    def send_kv_cache(
        self,
        args: KVTransferArgs,
        kv_data: List[torch.Tensor],
    ) -> None:
        with torch.cuda.stream(self.transfer_stream):
            sent_tensors = []
            for data in kv_data:
                contiguous_data = data.contiguous()
                sent_tensors.append(contiguous_data)
                dist.isend(tensor=contiguous_data, dst=args.dst_rank, group=self.transfer_group)
            event = torch.cuda.Event()
            event.record(self.transfer_stream)

        self.pending_transfers[args.uid] = (TransferStatus.TRANSFERRING, event, sent_tensors)

    def recv_kv_cache(self, args: KVTransferArgs, kv_buffers: List[torch.Tensor]) -> None:
        with torch.cuda.stream(self.transfer_stream):
            for buffer in kv_buffers:
                dist.irecv(tensor=buffer, src=args.src_rank, group=self.transfer_group)
            event = torch.cuda.Event()
            event.record(self.transfer_stream)

        self.pending_transfers[args.uid] = (TransferStatus.TRANSFERRING, event, [])

    def poll(self, uid: int) -> TransferStatus:
        entry = self.pending_transfers.get(uid)
        if entry is None:
            return TransferStatus.FAILED
        status, event, _ = entry
        if status in (TransferStatus.SUCCESS, TransferStatus.FAILED):
            return status
        if status != TransferStatus.TRANSFERRING:
            return status
        if event.query():
            self.pending_transfers[uid] = (TransferStatus.SUCCESS, event, _)
            return TransferStatus.SUCCESS
        return TransferStatus.TRANSFERRING

    def cleanup(self, uid: int) -> None:
        entry = self.pending_transfers.get(uid)
        if entry is not None:
            _, event, _ = entry
            if not event.query():
                event.synchronize()
            del self.pending_transfers[uid]

    def shutdown(self):
        for uid in list(self.pending_transfers.keys()):
            self.cleanup(uid)
        if self.transfer_group is not None:
            dist.destroy_process_group(self.transfer_group)
            self.transfer_group = None
