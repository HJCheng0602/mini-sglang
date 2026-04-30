"""Tests for DecodeWorker."""
from __future__ import annotations

import os
from unittest.mock import patch

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.pd.config import PDConfig
from minisgl.pd.decodeworker import DecodeWorker
from minisgl.pd.message import PrefillDoneMsg
from minisgl.pd.transfer import TransferStatus
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_test_config():
    return PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role="decode",
        use_dummy_weight=True,
        max_extend_tokens=256,
        max_running_req=4,
        kv_transfer_backend="nccl",
    )


@call_if_main()
def test_decode_worker():
    """All DecodeWorker tests in one function."""
    config = _create_test_config()
    worker = DecodeWorker(config)

    # 1: Init
    assert worker.scheduler is not None
    assert worker.kv_transfer is not None
    assert worker.io is not None
    assert len(worker.pending_transfers) == 0
    logger.info("test_decode_worker_init passed")

    # 2: Handle prefill done (mock KV recv)
    device_len = 50
    msg = PrefillDoneMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (device_len,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
        cached_len=0,
        device_len=device_len,
        kv_cache_indices=torch.arange(device_len, dtype=torch.int32),
        src_rank=0,
    )

    with patch.object(worker.kv_transfer, 'recv') as mock_recv:
        worker._handle_prefill_done(msg)
        assert mock_recv.called
    assert 0 in worker.pending_transfers
    logger.info("test_decode_worker_handle_prefill_done passed")

    # 3: Simulate KV transfer completion
    # Manually write random data into the local KV cache at the allocated pages
    _, local_page_indices = worker.pending_transfers[0]
    kv_cache = worker.scheduler.engine.kv_cache
    for layer_id in range(kv_cache.num_layers):
        k = kv_cache.k_cache(layer_id)
        v = kv_cache.v_cache(layer_id)
        k[local_page_indices[:device_len]] = torch.randn_like(k[local_page_indices[:device_len]])
        v[local_page_indices[:device_len]] = torch.randn_like(v[local_page_indices[:device_len]])

    # Mock poll to return SUCCESS
    with patch.object(worker.kv_transfer, 'poll', return_value=TransferStatus.SUCCESS), \
         patch.object(worker.kv_transfer, 'cleanup'):
        worker._check_pending_transfers()

    assert 0 not in worker.pending_transfers
    assert worker.scheduler.has_running_reqs()
    logger.info("test_decode_worker_check_pending_transfers passed")

    # 4: Run decode loop iteration (mock send_detokenize to capture output)
    with patch.object(worker.io, 'send_detokenize') as mock_send:
        worker._loop_iteration()
        assert mock_send.called
        msgs = mock_send.call_args[0][0]
        assert len(msgs) == 1
        assert msgs[0].uid == 0
    logger.info("test_decode_worker_loop_iteration passed")

    # 5: Multiple requests
    for i in range(1, 3):
        msg = PrefillDoneMsg(
            uid=i,
            input_ids=torch.randint(0, 1000, (device_len,), dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=5),
            cached_len=0,
            device_len=device_len,
            kv_cache_indices=torch.arange(device_len, dtype=torch.int32),
            src_rank=0,
        )
        with patch.object(worker.kv_transfer, 'recv'):
            worker._handle_prefill_done(msg)

    # Simulate transfers completing
    for uid in list(worker.pending_transfers.keys()):
        _, local_pages = worker.pending_transfers[uid]
        for layer_id in range(kv_cache.num_layers):
            kv_cache.k_cache(layer_id)[local_pages[:device_len]] = torch.randn_like(
                kv_cache.k_cache(layer_id)[local_pages[:device_len]]
            )
            kv_cache.v_cache(layer_id)[local_pages[:device_len]] = torch.randn_like(
                kv_cache.v_cache(layer_id)[local_pages[:device_len]]
            )

    with patch.object(worker.kv_transfer, 'poll', return_value=TransferStatus.SUCCESS), \
         patch.object(worker.kv_transfer, 'cleanup'):
        worker._check_pending_transfers()

    assert worker.scheduler.has_running_reqs()
    logger.info("test_decode_worker_multiple_requests passed")

    # 6: Shutdown
    worker.shutdown()
    logger.info("All DecodeWorker tests passed")
