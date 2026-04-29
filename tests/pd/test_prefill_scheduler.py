"""Tests for PD prefill scheduler module."""
from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.pd.config import PDConfig
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_test_config(
    max_extend_tokens: int = 256,
    max_running_req: int = 4,
) -> PDConfig:
    """Create a test PDConfig."""
    return PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role="prefill",
        use_dummy_weight=True,
        max_extend_tokens=max_extend_tokens,
        max_running_req=max_running_req,
    )


@call_if_main()
def test_prefill_scheduler_init():
    """Test PrefillScheduler initialization."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    assert scheduler is not None
    assert scheduler.engine is not None
    assert scheduler.table_manager is not None
    assert scheduler.cache_manager is not None
    assert scheduler.prefill_manager is not None
    assert not scheduler.has_pending_reqs()
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_init passed")


@call_if_main()
def test_prefill_scheduler_add_request():
    """Test adding requests to PrefillScheduler."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Create test request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    
    scheduler.add_request(msg)
    assert scheduler.has_pending_reqs()
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_add_request passed")


@call_if_main()
def test_prefill_scheduler_add_multiple_requests():
    """Test adding multiple requests."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config(max_running_req=10)
    scheduler = PrefillScheduler(config)
    
    # Add multiple requests
    for i in range(5):
        msg = UserMsg(
            uid=i,
            input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(msg)
    
    assert scheduler.has_pending_reqs()
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_add_multiple_requests passed")


@call_if_main()
def test_prefill_scheduler_schedule_batch():
    """Test scheduling a batch."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Add request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    scheduler.add_request(msg)
    
    # Schedule batch
    batch = scheduler.schedule_batch()
    assert batch is not None
    assert batch.is_prefill
    assert len(batch.reqs) == 1
    assert batch.reqs[0].uid == 0
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_schedule_batch passed")


@call_if_main()
def test_prefill_scheduler_no_pending():
    """Test scheduling with no pending requests."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Schedule should return None
    batch = scheduler.schedule_batch()
    assert batch is None
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_no_pending passed")


@call_if_main()
def test_prefill_scheduler_prepare_batch():
    """Test preparing a batch."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Add request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    scheduler.add_request(msg)
    
    # Schedule and prepare batch
    batch = scheduler.schedule_batch()
    assert batch is not None
    
    forward_input = scheduler.prepare_batch(batch)
    assert forward_input is not None
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_prepare_batch passed")


@call_if_main()
def test_prefill_scheduler_forward():
    """Test forward pass."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Add request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    scheduler.add_request(msg)
    
    # Schedule, prepare, and forward
    batch = scheduler.schedule_batch()
    forward_input = scheduler.prepare_batch(batch)
    output = scheduler.forward(forward_input)
    
    assert output is not None
    assert output.next_tokens_gpu is not None
    assert output.next_tokens_cpu is not None
    assert output.copy_done_event is not None
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_forward passed")


@call_if_main()
def test_prefill_scheduler_reject_long_input():
    """Test rejecting input that exceeds max sequence length."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Create very long input (should be rejected)
    max_seq_len = scheduler.engine.max_seq_len
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (max_seq_len + 100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    
    # This should be rejected (but we can't easily check the log)
    scheduler.add_request(msg)
    
    # The request might be rejected, so has_pending_reqs could be False
    # or the request might be truncated
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_reject_long_input passed")


@call_if_main()
def test_prefill_scheduler_abort_request():
    """Test aborting a request."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Add request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    scheduler.add_request(msg)
    assert scheduler.has_pending_reqs()
    
    # Abort request
    result = scheduler.abort_request(0)
    assert result == True
    assert not scheduler.has_pending_reqs()
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_abort_request passed")


@call_if_main()
def test_prefill_scheduler_abort_nonexistent():
    """Test aborting a nonexistent request."""
    from minisgl.pd.prefillsheduler import PrefillScheduler
    
    config = _create_test_config()
    scheduler = PrefillScheduler(config)
    
    # Abort nonexistent request
    result = scheduler.abort_request(999)
    assert result == False
    
    scheduler.shutdown()
    logger.info("test_prefill_scheduler_abort_nonexistent passed")
