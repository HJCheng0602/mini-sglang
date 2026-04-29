"""Tests for PD message module: serialization and deserialization."""
from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.pd.message import (
    PrefillDoneMsg,
    KVTransferReq,
    KVTransferAck,
    PrefillWorkerReady,
    DecodeWorkerReady,
)
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@call_if_main()
def test_prefill_done_msg():
    """Test PrefillDoneMsg serialization."""
    msg = PrefillDoneMsg(
        uid=123,
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0.8, max_tokens=100),
        cached_len=0,
        device_len=5,
        kv_cache_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        src_rank=0,
    )
    
    # Test serialization
    serialized = msg.encoder()
    assert serialized["__type__"] == "PrefillDoneMsg"
    assert serialized["uid"] == 123
    assert serialized["cached_len"] == 0
    assert serialized["device_len"] == 5
    assert serialized["src_rank"] == 0
    
    # Test deserialization
    deserialized = PrefillDoneMsg.decoder(serialized)
    assert deserialized.uid == msg.uid
    assert deserialized.cached_len == msg.cached_len
    assert deserialized.device_len == msg.device_len
    assert deserialized.src_rank == msg.src_rank
    assert torch.equal(deserialized.input_ids, msg.input_ids)
    assert torch.equal(deserialized.kv_cache_indices, msg.kv_cache_indices)
    assert deserialized.sampling_params.temperature == msg.sampling_params.temperature
    assert deserialized.sampling_params.max_tokens == msg.sampling_params.max_tokens
    
    logger.info("test_prefill_done_msg passed")


@call_if_main()
def test_kv_transfer_req():
    """Test KVTransferReq serialization."""
    msg = KVTransferReq(
        uid=456,
        src_rank=0,
        dst_rank=1,
        num_pages=10,
        paged_indices=torch.arange(10, dtype=torch.int32),
        chunk_id=0,
        is_last_chunk=True,
    )
    
    # Test serialization
    serialized = msg.encoder()
    assert serialized["__type__"] == "KVTransferReq"
    assert serialized["uid"] == 456
    assert serialized["src_rank"] == 0
    assert serialized["dst_rank"] == 1
    assert serialized["num_pages"] == 10
    assert serialized["chunk_id"] == 0
    assert serialized["is_last_chunk"] == True
    
    # Test deserialization
    deserialized = KVTransferReq.decoder(serialized)
    assert deserialized.uid == msg.uid
    assert deserialized.src_rank == msg.src_rank
    assert deserialized.dst_rank == msg.dst_rank
    assert deserialized.num_pages == msg.num_pages
    assert deserialized.chunk_id == msg.chunk_id
    assert deserialized.is_last_chunk == msg.is_last_chunk
    assert torch.equal(deserialized.paged_indices, msg.paged_indices)
    
    logger.info("test_kv_transfer_req passed")


@call_if_main()
def test_kv_transfer_ack():
    """Test KVTransferAck serialization."""
    msg = KVTransferAck(
        uid=789,
        chunk_id=0,
        success=True,
        error_msg="",
    )
    
    # Test serialization
    serialized = msg.encoder()
    assert serialized["__type__"] == "KVTransferAck"
    assert serialized["uid"] == 789
    assert serialized["chunk_id"] == 0
    assert serialized["success"] == True
    assert serialized["error_msg"] == ""
    
    # Test deserialization
    deserialized = KVTransferAck.decoder(serialized)
    assert deserialized.uid == msg.uid
    assert deserialized.chunk_id == msg.chunk_id
    assert deserialized.success == msg.success
    assert deserialized.error_msg == msg.error_msg
    
    logger.info("test_kv_transfer_ack passed")


@call_if_main()
def test_kv_transfer_ack_with_error():
    """Test KVTransferAck with error message."""
    msg = KVTransferAck(
        uid=100,
        chunk_id=1,
        success=False,
        error_msg="Connection timeout",
    )
    
    # Test serialization
    serialized = msg.encoder()
    
    # Test deserialization
    deserialized = KVTransferAck.decoder(serialized)
    assert deserialized.uid == 100
    assert deserialized.chunk_id == 1
    assert deserialized.success == False
    assert deserialized.error_msg == "Connection timeout"
    
    logger.info("test_kv_transfer_ack_with_error passed")


@call_if_main()
def test_prefill_worker_ready():
    """Test PrefillWorkerReady serialization."""
    msg = PrefillWorkerReady(
        worker_rank=0,
        worker_endpoint="localhost:29500",
    )
    
    # Test serialization
    serialized = msg.encoder()
    assert serialized["__type__"] == "PrefillWorkerReady"
    assert serialized["worker_rank"] == 0
    assert serialized["worker_endpoint"] == "localhost:29500"
    
    # Test deserialization
    deserialized = PrefillWorkerReady.decoder(serialized)
    assert deserialized.worker_rank == 0
    assert deserialized.worker_endpoint == "localhost:29500"
    
    logger.info("test_prefill_worker_ready passed")


@call_if_main()
def test_decode_worker_ready():
    """Test DecodeWorkerReady serialization."""
    msg = DecodeWorkerReady(
        worker_rank=1,
        worker_endpoint="localhost:29600",
    )
    
    # Test serialization
    serialized = msg.encoder()
    assert serialized["__type__"] == "DecodeWorkerReady"
    assert serialized["worker_rank"] == 1
    assert serialized["worker_endpoint"] == "localhost:29600"
    
    # Test deserialization
    deserialized = DecodeWorkerReady.decoder(serialized)
    assert deserialized.worker_rank == 1
    assert deserialized.worker_endpoint == "localhost:29600"
    
    logger.info("test_decode_worker_ready passed")


@call_if_main()
def test_message_roundtrip():
    """Test multiple serialization/deserialization roundtrips."""
    msg = PrefillDoneMsg(
        uid=42,
        input_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0.5, top_k=50, max_tokens=200),
        cached_len=0,
        device_len=3,
        kv_cache_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        src_rank=0,
    )
    
    # Multiple roundtrips
    for _ in range(5):
        serialized = msg.encoder()
        deserialized = PrefillDoneMsg.decoder(serialized)
        msg = deserialized
    
    # Should still have same values
    assert msg.uid == 42
    assert msg.cached_len == 0
    assert msg.device_len == 3
    assert torch.equal(msg.input_ids, torch.tensor([10, 20, 30], dtype=torch.int32))
    
    logger.info("test_message_roundtrip passed")
