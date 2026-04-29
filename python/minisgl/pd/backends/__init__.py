from __future__ import annotations
from minisgl.utils import Registry
from typing import Protocol
from ..base import BaseKVTransferBackend

class TransferBackendCreator(Protocol):
    def __call__(self, backend_config: dict) -> BaseKVTransferBackend: ...

SUPPORTED_TRANSFER_BACKENDS = Registry[TransferBackendCreator]('transfer_backends')

@SUPPORTED_TRANSFER_BACKENDS.register('nccl')
def create_nccl_backend(backend_config: dict) -> BaseKVTransferBackend:
    from .nccl_backend import NCCLTransferBackend
    return NCCLTransferBackend(backend_config)

@SUPPORTED_TRANSFER_BACKENDS.register('gloo')
def create_gloo_backend(backend_config: dict) -> BaseKVTransferBackend:
    from .gloo_backend import GlooTransferBackend
    return GlooTransferBackend(backend_config)

def create_transfer_backend(backend_name: str, backend_config: dict) -> BaseKVTransferBackend:
    return SUPPORTED_TRANSFER_BACKENDS[backend_name](backend_config)

__all__ = [
    "BaseKVTransferBackend",
    "SUPPORTED_TRANSFER_BACKENDS",
    "create_transfer_backend"
]