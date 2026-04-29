"""Tests for PD prefill IO module."""
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
    )


@call_if_main()
def test_prefill_io_init():
    """Test PrefillWorkerIO initialization."""
    from minisgl.pd.prefillio import PrefillWorkerIO
    
    config = _create_test_config()
    io = PrefillWorkerIO(config)
    
    assert io is not None
    assert io.config == config
    assert io.recv_from_tokenizer is not None
    assert io.send_to_decode is not None
    
    io.shutdown()
    logger.info("test_prefill_io_init passed")


@call_if_main()
def test_prefill_io_recv_messages_empty():
    """Test receiving messages when empty."""
    from minisgl.pd.prefillio import PrefillWorkerIO
    
    config = _create_test_config()
    io = PrefillWorkerIO(config)
    
    # Should return empty list
    msgs = io.recv_messages()
    assert isinstance(msgs, list)
    assert len(msgs) == 0
    
    io.shutdown()
    logger.info("test_prefill_io_recv_messages_empty passed")


@call_if_main()
def test_prefill_io_shutdown():
    """Test PrefillWorkerIO shutdown."""
    from minisgl.pd.prefillio import PrefillWorkerIO
    
    config = _create_test_config()
    io = PrefillWorkerIO(config)
    
    # Should not raise
    io.shutdown()
    
    logger.info("test_prefill_io_shutdown passed")
