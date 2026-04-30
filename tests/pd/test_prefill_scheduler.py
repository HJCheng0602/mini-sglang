"""Tests for PD prefill scheduler module."""
from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.prefillsheduler import PrefillScheduler
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_test_config(max_extend_tokens=256, max_running_req=4):
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
def test_prefill_scheduler():
    """All PrefillScheduler tests in one function (single Engine instance)."""
    config = _create_test_config()
    scheduler = PrefillScheduler(config)

    # 1: Init
    assert scheduler.engine is not None
    assert scheduler.table_manager is not None
    assert scheduler.cache_manager is not None
    assert scheduler.prefill_manager is not None
    assert not scheduler.has_pending_reqs()
    logger.info("test_prefill_scheduler_init passed")

    # 2: Add request
    msg = UserMsg(
        uid=0,
        input_ids=torch.randint(0, 1000, (100,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=50),
    )
    scheduler.add_request(msg)
    assert scheduler.has_pending_reqs()
    logger.info("test_prefill_scheduler_add_request passed")

    # 3: Schedule batch
    batch = scheduler.schedule_batch()
    assert batch is not None
    assert batch.is_prefill
    assert len(batch.reqs) == 1
    assert batch.reqs[0].uid == 0
    logger.info("test_prefill_scheduler_schedule_batch passed")

    # 4: Prepare batch
    forward_input = scheduler.prepare_batch(batch)
    assert forward_input is not None
    logger.info("test_prefill_scheduler_prepare_batch passed")

    # 5: Forward
    output = scheduler.forward(forward_input)
    assert output is not None
    assert output.next_tokens_gpu is not None
    assert output.next_tokens_cpu is not None
    assert output.copy_done_event is not None
    logger.info("test_prefill_scheduler_forward passed")

    # 6: No pending after schedule
    assert not scheduler.has_pending_reqs()
    batch = scheduler.schedule_batch()
    assert batch is None
    logger.info("test_prefill_scheduler_no_pending passed")

    # 7: Add multiple requests
    for i in range(1, 4):
        msg = UserMsg(
            uid=i,
            input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(msg)
    assert scheduler.has_pending_reqs()
    batch = scheduler.schedule_batch()
    assert batch is not None and len(batch.reqs) >= 1
    logger.info("test_prefill_scheduler_add_multiple_requests passed")

    # 8: Abort a pending request
    msg = UserMsg(
        uid=99,
        input_ids=torch.randint(0, 1000, (50,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
    )
    scheduler.add_request(msg)
    assert scheduler.has_pending_reqs()
    result = scheduler.abort_request(99)
    assert result == True
    result = scheduler.abort_request(999)
    assert result == False
    logger.info("test_prefill_scheduler_abort_request passed")

    # 9: End-to-end with page indices
    # Free resources from step 7's batch first
    for req in batch.reqs:
        scheduler._free_req_resources(req)
    msg = UserMsg(
        uid=10,
        input_ids=torch.randint(0, 1000, (80,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=20),
    )
    scheduler.add_request(msg)
    batch = scheduler.schedule_batch()
    assert batch is not None, "schedule_batch returned None for uid=10"
    forward_input = scheduler.prepare_batch(batch)
    output = scheduler.forward(forward_input)
    req = batch.reqs[0]
    page_indices = scheduler.engine.page_table[req.table_idx, :req.device_len]
    assert len(page_indices) == req.device_len
    logger.info("test_prefill_scheduler_end_to_end passed")

    scheduler.shutdown()
    logger.info("All PrefillScheduler tests passed")
