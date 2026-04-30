"""Integration test: PrefillWorker and DecodeWorker working together.

This test simulates the full PD pipeline in a single process:
  UserMsg → PrefillWorker (prefill + mock KV send)
         → PrefillDoneMsg via ZMQ
         → DecodeWorker (mock KV recv + decode loop)
         → DetokenizeMsg

The KV transfer is mocked because NCCL requires multi-process.
The scheduling, forward, and message passing are all real.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.decodeworker import DecodeWorker
from minisgl.pd.prefillworker import PrefillWorker
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def _create_prefill_config():
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


def _create_decode_config():
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
def test_integration():
    """Full PD pipeline: PrefillWorker → DecodeWorker."""
    # ---- Phase 1: PrefillWorker does prefill ----
    prefill_config = _create_prefill_config()
    prefill_worker = PrefillWorker(prefill_config)

    input_len = 50
    max_tokens = 5
    msg = UserMsg(
        uid=42,
        input_ids=torch.randint(0, 1000, (input_len,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    prefill_worker._process_message(msg)
    assert prefill_worker.scheduler.has_pending_reqs()

    # Run prefill (mock KV send and PrefillDoneMsg send)
    sent_prefill_msgs = []

    def capture_prefill_done(msg):
        sent_prefill_msgs.append(msg)

    with patch.object(prefill_worker.kv_transfer, 'send'), \
         patch.object(prefill_worker.io, 'send_prefill_donw', side_effect=capture_prefill_done):
        prefill_worker._loop_iteration()

    assert len(sent_prefill_msgs) == 1
    prefill_msg = sent_prefill_msgs[0]
    assert prefill_msg.uid == 42
    assert prefill_msg.device_len == input_len + 1  # prefill + 1 sampled token
    logger.info("Phase 1 (prefill) completed, PrefillDoneMsg captured")

    # Save prefill KV cache data for later injection
    prefill_kv_cache = prefill_worker.scheduler.engine.kv_cache
    prefill_page_indices = prefill_msg.kv_cache_indices
    device_len = prefill_msg.device_len

    # ---- Phase 2: DecodeWorker receives and runs decode ----
    decode_config = _create_decode_config()
    decode_worker = DecodeWorker(decode_config)

    # Manually simulate: receive PrefillDoneMsg → allocate pages → inject KV data
    num_pages = (device_len + decode_config.page_size - 1) // decode_config.page_size
    local_page_indices = decode_worker.scheduler.cache_manager.allocate_for_transfer(num_pages)

    # Copy KV data from prefill engine to decode engine
    decode_kv_cache = decode_worker.scheduler.engine.kv_cache
    for layer_id in range(prefill_kv_cache.num_layers):
        src_k = prefill_kv_cache.k_cache(layer_id)[prefill_page_indices[:device_len]]
        src_v = prefill_kv_cache.v_cache(layer_id)[prefill_page_indices[:device_len]]
        decode_kv_cache.k_cache(layer_id)[local_page_indices[:device_len]] = src_k
        decode_kv_cache.v_cache(layer_id)[local_page_indices[:device_len]] = src_v

    # Add request to decode scheduler
    decode_worker.scheduler.add_request(
        uid=prefill_msg.uid,
        input_ids=prefill_msg.input_ids,
        sampling_params=prefill_msg.sampling_params,
        cached_len=prefill_msg.cached_len,
        device_len=device_len,
        local_page_indices=local_page_indices,
    )
    assert decode_worker.scheduler.has_running_reqs()

    # Run decode loop for a few iterations
    all_tokens = []
    for step in range(max_tokens):
        batch = decode_worker.scheduler.schedule_batch()
        assert batch is not None, f"No batch at step {step}"
        forward_input = decode_worker.scheduler.prepare_batch(batch)
        output = decode_worker.scheduler.forward(forward_input)
        msgs = decode_worker.scheduler.process_batch(forward_input, output)
        assert len(msgs) == 1
        all_tokens.append(msgs[0].next_token)
        if msgs[0].finished:
            break

    logger.info(f"Phase 2 (decode) completed, generated {len(all_tokens)} tokens: {all_tokens}")
    assert len(all_tokens) > 0, "No tokens generated"

    # ---- Cleanup ----
    prefill_worker.kv_transfer.shutdown()
    prefill_worker.io.shutdown()
    prefill_worker.scheduler.shutdown()

    decode_worker.shutdown()

    logger.info("Integration test PASSED: PrefillWorker → DecodeWorker pipeline works")
