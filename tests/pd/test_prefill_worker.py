"""Tests for PD prefill worker module."""
from __future__ import annotations

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import ExitMsg, UserMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.prefillworker import PrefillWorker
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_test_config():
    return PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role="prefill",
        use_dummy_weight=True,
        max_extend_tokens=256,
        max_running_req=4,
        kv_transfer_backend="nccl",
    )


@call_if_main()
def test_prefill_worker():
    """All PrefillWorker tests in one function (single Engine instance)."""
    config = _create_test_config()
    worker = PrefillWorker(config)

    # 1: Init
    assert worker.config == config
    assert worker.scheduler is not None
    assert worker.kv_transfer is not None
    assert worker.io is not None
    logger.info("test_prefill_worker_init passed")

    # 2: Process UserMsg
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    worker._process_message(msg)
    assert worker.scheduler.has_pending_reqs()
    logger.info("test_prefill_worker_process_user_msg passed")

    # 3: Process ExitMsg raises KeyboardInterrupt
    try:
        worker._process_message(ExitMsg())
        assert False, "Should have raised KeyboardInterrupt"
    except KeyboardInterrupt:
        pass
    logger.info("test_prefill_worker_process_exit_msg passed")

    # 4: Loop iteration (mock KV transfer, only test scheduling + forward)
    msg = UserMsg(
        uid=1,
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    worker.scheduler.add_request(msg)
    with patch.object(worker.kv_transfer, 'send'), \
         patch.object(worker.io, 'send_prefill_donw'):
        worker._loop_iteration()
    logger.info("test_prefill_worker_loop_iteration passed")

    # 5: Get page indices
    msg = UserMsg(
        uid=2,
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    worker.scheduler.add_request(msg)
    batch = worker.scheduler.schedule_batch()
    assert batch is not None and len(batch.reqs) > 0
    req = batch.reqs[0]
    page_indices = worker._get_page_indices(req)
    assert page_indices is not None
    assert len(page_indices) == req.device_len
    logger.info("test_prefill_worker_get_page_indices passed")

    # 6: Select decode worker
    from minisgl.core import Req
    mock_req = Req(
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        table_idx=0, cached_len=0, output_len=10, uid=99,
        sampling_params=SamplingParams(max_tokens=10), cache_handle=None,
    )
    dst_rank = worker._select_decode_worker(mock_req)
    assert isinstance(dst_rank, int) and dst_rank >= 0
    logger.info("test_prefill_worker_select_decode_worker passed")

    # 7: End-to-end loop iteration (mock transfer, verify full pipeline)
    msg = UserMsg(
        uid=3,
        input_ids=torch.randint(0, 1000, (60,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    worker._process_message(msg)
    with patch.object(worker.kv_transfer, 'send') as mock_send, \
         patch.object(worker.io, 'send_prefill_donw') as mock_done:
        worker._loop_iteration()
        assert mock_send.called, "kv_transfer.send was not called"
        assert mock_done.called, "io.send_prefill_donw was not called"
    logger.info("test_prefill_worker_end_to_end passed")

    worker.shutdown()
    logger.info("All PrefillWorker tests passed")
