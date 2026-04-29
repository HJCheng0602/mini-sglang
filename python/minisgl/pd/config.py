from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minisgl.scheduler import SchedulerConfig

@dataclass(frozen=True)
class PDConfig(SchedulerConfig):

    pd_enabled: bool = False
    role: Literal["prefill", "decode", "standalone"] = "standalone"

    prefill_worker_addr: str = "localhost"
    prefill_worker_port: int = 29500

    decode_worker_addr: str = "localhost"
    decode_worker_port: int = 29000

    kv_transfer_backend: Literal["nccl", "gloo"] = "nccl"
    kv_transfer_backend_config: dict = None

    kv_transfer_chunk_size: int = 0  # 0 means no chunking
    kv_transfer_overlap: bool = True

    router_strategy: Literal["round_robin", "load_balance"] = "round_robin"

    pd_debug: bool = False

    def __post_init__(self):
        if self.kv_transfer_backend_config is None:
            object.__setattr__(self, 'kv_transfer_backend_config', {})

    @property
    def is_prefill_worker(self) -> bool:
        return self.pd_enabled and self.role == "prefill"
    
    @property
    def is_decode_worker(self) -> bool:
        return self.pd_enabled and self.role == "decode"
    
    @property
    def is_standalone(self) -> bool:
        return not self.pd_enabled or self.role == "standalone"

    