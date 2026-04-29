from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import torch

class TransferStatus(IntEnum):
    FAILED = 0
    BOOTSTRAPPING = 1
    WAITING_FOR_INPUT = 2
    TRANSFERRING = 3
    SUCCESS = 4


@dataclass
class KVTransferArgs:
    uid: int
    src_rank: int
    dst_rank: int

    num_layers: int
    kv_heads: int
    head_dim: int
    page_size: int
    num_pages: int

    kv_data_ptrs: Optional[List[int]] = None
    kv_data_lens: Optional[List[int]] = None

    chunk_size: int = 0


class BaseKVTransferBackend(ABC):

    @abstractmethod
    def __init__(self, backend_config: dict):
        pass

    @abstractmethod
    def init_transfer(self, args:KVTransferArgs) -> None:
        pass

    @abstractmethod
    def send_kv_cache(
        self,
        args:KVTransferArgs,
        kv_data: List[torch.Tensor]
    ) -> None:
        pass

    @abstractmethod
    def recv_kv_cache(
        self,
        args:KVTransferArgs,
        kv_buffers: List[torch.Tensor]
    ) -> None:
        pass

    @abstractmethod
    def poll(self, uid: int) -> TransferStatus:
        pass

    @abstractmethod
    def cleanup(self, uid: int) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
         