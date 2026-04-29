"""Tests for PD prefill worker module."""
from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.pd.config import PDConfig
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_test_config() -> PDConfig:
    """Create a test PDConfig."""
    return PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role="prefill",
        use_dummy_weight=True,
        max_extend_tokens=256,
        max_running_req=4,
    )


@call_if_main()
def test_prefill_worker_init():
    """Test PrefillWorker initialization."""
    from minisgl.pd.prefillworker import PrefillWorker
    
    config = _create_test_config()
    worker = PrefillWorker(config)
    
    assert worker is not None
    assert worker.config == config
    assert worker.scheduler is not None
    assert worker.kv_transfer is not None
    assert worker.io is not None
    
    worker.shutdown()
    logger.info("test_prefill_worker_init passed")


@call_if_main()
def test_prefill_worker_process_user_msg():
    """Test processing UserMsg."""
    from minisgl.pd.prefillworker import PrefillWorker
    from minisgl.message import UserMsg
    
    config = _create_test_config()
    worker = PrefillWorker(config)
    
    # Create UserMsg
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    
    # Process message
    worker._process_message(msg)
    
    # Should have pending request
    assert worker.scheduler.has_pending_reqs()
    
    worker.shutdown()
    logger.info("test_prefill_worker_process_user_msg passed")


@call_if_main()
def test_prefill_worker_process_exit_msg():
    """Test processing ExitMsg raises KeyboardInterrupt."""
    from minisgl.pd.prefillworker import PrefillWorker
    from minisgl.message import ExitMsg
    
    config = _create_test_config()
    worker = PrefillWorker(config)
    
    # Process ExitMsg should raise KeyboardInterrupt
    try:
        worker._process_message(ExitMsg())
        assert False, "Should have raised KeyboardInterrupt"
    except KeyboardInterrupt:
        pass
    
    worker.shutdown()
    logger.info("test_prefill_worker_process_exit_msg passed")


@call_if_main()
def test_prefill_worker_loop_iteration():
    """Test single loop iteration."""
    from minisgl.pd.prefillworker import PrefillWorker
    from minisgl.message import UserMsg
    
    config = _create_test_config()
    worker = PrefillWorker(config)
    
    # Add a request by calling scheduler directly
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    worker.scheduler.add_request(msg)
    
    # Run one iteration
    worker._loop_iteration()
    
    # After processing, pending requests should be empty
    # (request should have been scheduled and processed)
    
    worker.shutdown()
    logger.info("test_prefill_worker_loop_iteration passed")


@call_if_main()
def test_prefill_worker_get_page_indices():
    """Test getting page indices."""
    from minisgl.pd.prefillworker import PrefillWorker
    from minisgl.message import UserMsg
    from minisgl.core import Req
    
    config = _create_test_config()
    worker = PrefillWorker(config)
    
    # Create a mock request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    worker.scheduler.add_request(msg)
    
    # Schedule to get a Req object
    batch = worker.scheduler.schedule_batch()
    assert batch is not None
    assert len(batch.reqs) > 0
    
    req = batch.reqs[0]
    page_indices = worker._get_page_indices(req)
    
    assert page_indices is not None
    assert len(page_indices) == req.device_len
    
    worker.shutdown()
    logger.info("test_prefill_worker_get_page_indices passed")


@call_if_main()
def test_prefill_worker_select_decode_worker():
    """Test selecting decode worker."""
    from minisgl.pd.prefillworker import PrefillWorker
    from minisgl.core import Req
    
    config = _create_test_config()
    worker = PrefillWorker(config)
    
    # Create a mock request
    req = Req(
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=10,
        uid=0,
        sampling_params=SamplingParams(max_tokens=10),
        cache_handle=None,
    )
    
    dst_rank = worker._select_decode_worker(req)
    assert isinstance(dst_rank, int)
    assert dst_rank >= 0
    
    worker.shutdown()
    logger.info("test_prefill_worker_select_decode_worker passed")
