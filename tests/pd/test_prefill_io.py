"""Tests for PD prefill IO module."""
from __future__ import annotations

import torch
from minisgl.distributed import DistributedInfo
from minisgl.pd.config import PDConfig
from minisgl.pd.prefillio import PrefillWorkerIO
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@call_if_main()
def test_prefill_io():
    """All PrefillWorkerIO tests in one function (single ZMQ binding)."""
    config = PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role="prefill",
        use_dummy_weight=True,
    )
    io = PrefillWorkerIO(config)

    # 1: Init
    assert io.config == config
    assert io.recv_from_tokenizer is not None
    assert io.send_to_decode is not None
    logger.info("test_prefill_io_init passed")

    # 2: Recv messages when empty
    msgs = io.recv_messages()
    assert isinstance(msgs, list)
    assert len(msgs) == 0
    logger.info("test_prefill_io_recv_messages_empty passed")

    # 3: Shutdown
    io.shutdown()
    logger.info("test_prefill_io_shutdown passed")

    logger.info("All PrefillWorkerIO tests passed")
