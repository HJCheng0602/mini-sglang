from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.utils import init_logger

from .backends import create_transfer_backend
from .base import BaseKVTransferBackend, KVTransferArgs, TransferStatus

logger = init_logger(__name__)


class KVTransferManager:
    def __init__(self, backend_name: str, config: dict):
        self.backend: BaseKVTransferBackend = create_transfer_backend(backend_name, config)
        self.backend_name = backend_name
        logger.info(f"KV Transfer Manager created with backend {backend_name}")

    def prepare_kv_data(
            self,
            kv_cache: MHAKVCache,
            paged_indices:torch.Tensor) -> List[torch.Tensor]:
        num_layers = kv_cache.num_layers
        kv_data_list = []

        for layer_id in range(num_layers):
            k_cache = kv_cache.k_cache(layer_id)
            v_cache = kv_cache.v_cache(layer_id)

            k_data = k_cache[paged_indices]
            v_data = v_cache[paged_indices]

            kv_data = torch.stack([k_data, v_data], dim=0)
            kv_data_list.append(kv_data)
        return kv_data_list
    
    def prepare_recv_buffers(
        self,
        kv_cache: MHAKVCache,
        num_pages: int,
        paged_indices: torch.Tensor
    ) -> List[torch.Tensor]:
        num_layers = kv_cache.num_layers
        page_size = kv_cache._kv_buffer.shape[3]
        head_dim = kv_cache._kv_buffer.shape[5]
        kv_heads = kv_cache._kv_buffer.shape[4]

        recv_buffers = []
        for layer_id in range(num_layers):
            buffer = torch.empty(
                (2, num_pages, page_size, kv_heads, head_dim),
                device=kv_cache.device,
                dtype=kv_cache.dtype,
            )
            recv_buffers.append(buffer)
        return recv_buffers
    
    def send(self, uid: int, kv_cache: MHAKVCache, paged_indices: torch.Tensor, dst_rank: int) -> None:
        kv_data_list = self.prepare_kv_data(kv_cache, paged_indices)

        args = KVTransferArgs(
            uid=uid,
            src_rank=dist.get_rank(),
            dst_rank=dst_rank,
            num_layers=kv_cache.num_layers,
            kv_heads=kv_cache._kv_buffer.shape[4],
            head_dim=kv_cache._kv_buffer.shape[5],
            page_size=kv_cache._kv_buffer.shape[3],
            num_pages=paged_indices.shape[0]
        )
        self.backend.init_transfer(args)
        self.backend.send_kv_cache(args, kv_data_list)

    def recv(self, uid: int, kv_cache: MHAKVCache, paged_indices: torch.Tensor, src_rank: int) -> None:
        num_pages = paged_indices.shape[0]
        recv_buffers = self.prepare_recv_buffers(kv_cache, num_pages, paged_indices)
        args = KVTransferArgs(
            uid=uid,
            src_rand=src_rank,
            dst_rank=dist.get_rank(),
            num_layers=kv_cache.num_layers,
            kv_heads=kv_cache._kv_buffer.shape[4],
            head_dim=kv_cache._kv_buffer.shape[5],
            page_size=kv_cache._kv_buffer.shape[3],
            num_pages=num_pages
        )
        self.backend.init_transfer(args)
        self.backend.recv_kv_cache(args, recv_buffers)

        page_indices_gpu = paged_indices.to(kv_cache.device, non_blocking=True)

        for layer_id, recv_buffer in enumerate(recv_buffers):
            k_data = recv_buffer[0]
            v_data = recv_buffer[1]

            kv_cache.k_cache(layer_id)[page_indices_gpu] = k_data
            kv_cache.v_cache(layer_id)[page_indices_gpu] = v_data

        logger.debug(f"Received KV cache for UID {uid} from rank {src_rank}")

    def poll(self, uid: int) -> TransferStatus:
        return self.backend.poll(uid)
    
    def cleanup(self, uid: int) -> None:
        self.backend.cleanup(uid)

    def shutdown(self):
        self.backend.shutdown()
        logger.info(f"KV Transfer Manager with backend {self.backend_name} has been shutdown")

    
