"""Tests for DecodeScheduler."""
from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.pd.config import PDConfig
from minisgl.pd.decodescheduler import DecodeScheduler
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_test_config(max_running_req=4):
    return PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role="decode",
        use_dummy_weight=True,
        max_running_req=max_running_req,
    )


@call_if_main()
def test_decode_scheduler():
    """All DecodeScheduler tests in one function."""
    config = _create_test_config()
    scheduler = DecodeScheduler(config)

    # 1: Init
    assert scheduler.engine is not None
    assert not scheduler.has_running_reqs()
    assert scheduler.schedule_batch() is None
    logger.info("test_decode_scheduler_init passed")

    # 2: Add a decode request (simulate receiving from prefill)
    # After prefill: input_len=50, sampled 1 token → device_len=51, cached_len=50
    cached_len = 50
    device_len = 51
    num_pages = device_len  # page_size=1
    local_page_indices = scheduler.cache_manager.allocate_for_transfer(num_pages)

    scheduler.add_request(
        uid=0,
        input_ids=torch.randint(0, 1000, (device_len,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=10),
        cached_len=cached_len,
        device_len=device_len,
        local_page_indices=local_page_indices,
    )
    assert scheduler.has_running_reqs()
    logger.info("test_decode_scheduler_add_request passed")

    # 3: Schedule decode batch
    batch = scheduler.schedule_batch()
    assert batch is not None
    assert batch.is_decode
    assert len(batch.reqs) == 1
    logger.info("test_decode_scheduler_schedule_batch passed")

    # 4: Prepare and forward
    forward_input = scheduler.prepare_batch(batch)
    output = scheduler.forward(forward_input)
    assert output.next_tokens_gpu is not None
    logger.info("test_decode_scheduler_forward passed")

    # 5: Process batch
    msgs = scheduler.process_batch(forward_input, output)
    assert len(msgs) == 1
    assert msgs[0].uid == 0
    assert isinstance(msgs[0].next_token, int)
    logger.info("test_decode_scheduler_process_batch passed")

    # 6: Add multiple requests
    for i in range(1, 4):
        indices = scheduler.cache_manager.allocate_for_transfer(num_pages)
        scheduler.add_request(
            uid=i,
            input_ids=torch.randint(0, 1000, (device_len,), dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=5),
            cached_len=cached_len,
            device_len=device_len,
            local_page_indices=indices,
        )
    batch = scheduler.schedule_batch()
    assert batch is not None and len(batch.reqs) == 4
    logger.info("test_decode_scheduler_multiple_requests passed")

    # 7: Abort request
    assert scheduler.abort_request(1) == True
    assert scheduler.abort_request(999) == False
    logger.info("test_decode_scheduler_abort passed")

    scheduler.shutdown()
    logger.info("All DecodeScheduler tests passed")
