"""Tests for PD transfer module: backends and KVTransferManager."""
from __future__ import annotations

import torch
from minisgl.pd.base import KVTransferArgs, TransferStatus
from minisgl.pd.backends import (
    SUPPORTED_TRANSFER_BACKENDS,
    create_transfer_backend,
)
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@call_if_main()
def test_backend_registry():
    """Test that backends are registered correctly."""
    supported = SUPPORTED_TRANSFER_BACKENDS.supported_names()
    assert "nccl" in supported, f"nccl not in supported backends: {supported}"
    assert "gloo" in supported, f"gloo not in supported backends: {supported}"
    
    logger.info(f"Supported backends: {supported}")
    logger.info("test_backend_registry passed")


@call_if_main()
def test_create_gloo_backend():
    """Test creating Gloo backend."""
    backend = create_transfer_backend("gloo", {})
    assert backend is not None
    assert backend.__class__.__name__ == "GlooTransferBackend"
    
    logger.info("test_create_gloo_backend passed")


@call_if_main()
def test_create_nccl_backend():
    """Test creating NCCL backend."""
    backend = create_transfer_backend("nccl", {})
    assert backend is not None
    assert backend.__class__.__name__ == "NCCLTransferBackend"
    
    logger.info("test_create_nccl_backend passed")


@call_if_main()
def test_create_invalid_backend():
    """Test creating invalid backend raises error."""
    try:
        backend = create_transfer_backend("invalid_backend", {})
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    logger.info("test_create_invalid_backend passed")


@call_if_main()
def test_gloo_backend_init_transfer():
    """Test Gloo backend init_transfer."""
    backend = create_transfer_backend("gloo", {})
    
    args = KVTransferArgs(
        uid=123,
        src_rank=0,
        dst_rank=1,
        num_layers=32,
        kv_heads=8,
        head_dim=128,
        page_size=16,
        num_pages=10,
    )
    
    # Should not raise
    backend.init_transfer(args)
    
    # Check status
    status = backend.poll(123)
    assert status == TransferStatus.BOOTSTRAPPING
    
    # Cleanup
    backend.cleanup(123)
    status = backend.poll(123)
    assert status == TransferStatus.FAILED
    
    logger.info("test_gloo_backend_init_transfer passed")


@call_if_main()
def test_gloo_backend_poll_nonexistent():
    """Test polling for nonexistent transfer."""
    backend = create_transfer_backend("gloo", {})
    
    status = backend.poll(999)
    assert status == TransferStatus.FAILED
    
    logger.info("test_gloo_backend_poll_nonexistent passed")


@call_if_main()
def test_gloo_backend_cleanup_nonexistent():
    """Test cleaning up nonexistent transfer."""
    backend = create_transfer_backend("gloo", {})
    
    # Should not raise
    backend.cleanup(999)
    
    logger.info("test_gloo_backend_cleanup_nonexistent passed")


@call_if_main()
def test_kv_transfer_args_creation():
    """Test KVTransferArgs creation with various parameters."""
    args = KVTransferArgs(
        uid=1,
        src_rank=0,
        dst_rank=1,
        num_layers=32,
        kv_heads=8,
        head_dim=128,
        page_size=16,
        num_pages=10,
        chunk_size=5,
    )
    
    assert args.uid == 1
    assert args.src_rank == 0
    assert args.dst_rank == 1
    assert args.num_layers == 32
    assert args.kv_heads == 8
    assert args.head_dim == 128
    assert args.page_size == 16
    assert args.num_pages == 10
    assert args.chunk_size == 5
    
    logger.info("test_kv_transfer_args_creation passed")


@call_if_main()
def test_transfer_status_ordering():
    """Test TransferStatus ordering."""
    assert TransferStatus.FAILED < TransferStatus.BOOTSTRAPPING
    assert TransferStatus.BOOTSTRAPPING < TransferStatus.WAITING_FOR_INPUT
    assert TransferStatus.WAITING_FOR_INPUT < TransferStatus.TRANSFERRING
    assert TransferStatus.TRANSFERRING < TransferStatus.SUCCESS
    
    logger.info("test_transfer_status_ordering passed")


@call_if_main()
def test_multiple_backends():
    """Test creating multiple backends."""
    backends = []
    for name in ["gloo", "gloo", "nccl", "nccl"]:
        backend = create_transfer_backend(name, {})
        backends.append(backend)
    
    assert len(backends) == 4
    assert backends[0].__class__.__name__ == "GlooTransferBackend"
    assert backends[1].__class__.__name__ == "GlooTransferBackend"
    assert backends[2].__class__.__name__ == "NCCLTransferBackend"
    assert backends[3].__class__.__name__ == "NCCLTransferBackend"
    
    logger.info("test_multiple_backends passed")
