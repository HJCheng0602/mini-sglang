"""Tests for PD base module: TransferStatus, KVTransferArgs, BaseKVTransferBackend."""
from __future__ import annotations

import torch
from minisgl.pd.base import KVTransferArgs, TransferStatus
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@call_if_main()
def test_transfer_status_enum():
    """Test TransferStatus enum values."""
    assert TransferStatus.FAILED == 0
    assert TransferStatus.BOOTSTRAPPING == 1
    assert TransferStatus.WAITING_FOR_INPUT == 2
    assert TransferStatus.TRANSFERRING == 3
    assert TransferStatus.SUCCESS == 4
    
    # Test that enum values are distinct
    values = [
        TransferStatus.FAILED,
        TransferStatus.BOOTSTRAPPING,
        TransferStatus.WAITING_FOR_INPUT,
        TransferStatus.TRANSFERRING,
        TransferStatus.SUCCESS,
    ]
    assert len(set(values)) == 5
    
    logger.info("test_transfer_status_enum passed")


@call_if_main()
def test_kv_transfer_args_basic():
    """Test KVTransferArgs basic creation."""
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
    
    assert args.uid == 123
    assert args.src_rank == 0
    assert args.dst_rank == 1
    assert args.num_layers == 32
    assert args.kv_heads == 8
    assert args.head_dim == 128
    assert args.page_size == 16
    assert args.num_pages == 10
    assert args.chunk_size == 0  # default value
    assert args.kv_data_ptrs is None  # default value
    assert args.kv_data_lens is None  # default value
    
    logger.info("test_kv_transfer_args_basic passed")


@call_if_main()
def test_kv_transfer_args_with_ptrs():
    """Test KVTransferArgs with data pointers."""
    args = KVTransferArgs(
        uid=456,
        src_rank=0,
        dst_rank=1,
        num_layers=32,
        kv_heads=8,
        head_dim=128,
        page_size=16,
        num_pages=10,
        kv_data_ptrs=[0x1000, 0x2000],
        kv_data_lens=[1024, 2048],
        chunk_size=5,
    )
    
    assert args.uid == 456
    assert args.kv_data_ptrs == [0x1000, 0x2000]
    assert args.kv_data_lens == [1024, 2048]
    assert args.chunk_size == 5
    
    logger.info("test_kv_transfer_args_with_ptrs passed")


@call_if_main()
def test_kv_transfer_args_edge_cases():
    """Test KVTransferArgs edge cases."""
    # Test with uid=0
    args = KVTransferArgs(
        uid=0,
        src_rank=0,
        dst_rank=0,
        num_layers=1,
        kv_heads=1,
        head_dim=64,
        page_size=1,
        num_pages=1,
    )
    assert args.uid == 0
    assert args.num_pages == 1
    
    # Test with large values
    args = KVTransferArgs(
        uid=999999,
        src_rank=0,
        dst_rank=7,
        num_layers=80,
        kv_heads=64,
        head_dim=128,
        page_size=64,
        num_pages=10000,
    )
    assert args.uid == 999999
    assert args.dst_rank == 7
    assert args.num_layers == 80
    assert args.num_pages == 10000
    
    logger.info("test_kv_transfer_args_edge_cases passed")


@call_if_main()
def test_base_kv_transfer_backend_interface():
    """Test that BaseKVTransferBackend is abstract."""
    from minisgl.pd.base import BaseKVTransferBackend
    
    # Cannot instantiate abstract class
    try:
        backend = BaseKVTransferBackend({})
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass
    
    logger.info("test_base_kv_transfer_backend_interface passed")
