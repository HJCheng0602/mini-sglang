"""Real end-to-end PD test with NCCL KV transfer.

Usage:
    PYTHONPATH=python python tests/pd/test_e2e.py

Requires 2 GPUs. Tests the full pipeline:
  Rank 0 (PrefillWorker): UserMsg → prefill → NCCL send KV → ZMQ send PrefillDoneMsg
  Rank 1 (DecodeWorker):  ZMQ recv PrefillDoneMsg → NCCL recv KV → decode step → token
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from unittest.mock import patch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_python_dir = os.path.join(_root, "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

import torch
import torch.distributed as dist
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.decodeworker import DecodeWorker
from minisgl.pd.prefillworker import PrefillWorker
from minisgl.pd.transfer import TransferStatus
from minisgl.utils import init_logger

logger = init_logger(__name__)

# 2 free GPUs for this test
AVAILABLE_GPUS = [4, 5]
SHARED_SUFFIX = ".e2e_test"


def _run_worker(rank: int, role: str, ready_barrier, done_barrier):
    gpu_id = AVAILABLE_GPUS[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Initialize shared NCCL group BEFORE Engine (for KV transfer)
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    torch.cuda.set_device(0)

    # Patch torch.distributed so Engine doesn't conflict with our NCCL group:
    # - Engine._init_communication calls init_process_group again → mock it
    # - Engine._sync_get_memory calls all_reduce on CPU tensor with NCCL group → mock it
    # - Engine.__init__ asserts not torch.cuda.is_initialized → mock to bypass
    # - Engine.shutdown / nccl_backend.shutdown call destroy_process_group → mock
    #   (we handle the real destroy after all patches stop)
    patchers = [
        patch("torch.cuda.is_initialized", return_value=False),
        patch("torch.distributed.init_process_group"),
        patch("torch.distributed.destroy_process_group"),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_reduce"),
    ]
    for p in patchers:
        p.start()

    try:
        config = _get_config(role)
        if role == "prefill":
            worker = PrefillWorker(config)
        else:
            worker = DecodeWorker(config)

        if role == "prefill":
            _do_prefill(worker, done_barrier)
        else:
            _do_decode(worker, done_barrier)

        worker.kv_transfer.shutdown()
        worker.io.shutdown()
        worker.scheduler.shutdown()
    finally:
        for p in patchers:
            p.stop()

    # Destroy the real NCCL group (all mock'd destroys were no-ops)
    if dist.is_initialized():
        dist.destroy_process_group()


def _get_config(role: str) -> PDConfig:
    return PDConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        pd_enabled=True,
        role=role,
        use_dummy_weight=True,
        max_extend_tokens=256,
        max_running_req=4,
        kv_transfer_backend="nccl",
        _unique_suffix=SHARED_SUFFIX,
    )


def _do_prefill(worker, done_barrier):
    input_len = 50
    max_tokens = 5
    msg = UserMsg(
        uid=42,
        input_ids=torch.randint(0, 1000, (input_len,), dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    worker._process_message(msg)
    assert worker.scheduler.has_pending_reqs()

    # Wait for both workers to be ready
    done_barrier.wait()

    # Run prefill + NCCL send KV + ZMQ send PrefillDoneMsg
    worker._loop_iteration()
    logger.info("[rank 0] Prefill done, sent KV cache and PrefillDoneMsg")

    # Wait for rank 1 to finish
    done_barrier.wait()


def _do_decode(worker, done_barrier):
    # Wait for both workers to be ready
    done_barrier.wait()

    # Wait for PrefillDoneMsg from rank 0
    logger.info("[rank 1] Waiting for PrefillDoneMsg...")
    deadline = time.time() + 30
    prefill_msgs = []
    while time.time() < deadline:
        prefill_msgs = worker.io.recv_prefill_done()
        if prefill_msgs:
            break
        time.sleep(0.01)

    assert len(prefill_msgs) == 1, f"Expected 1 PrefillDoneMsg, got {len(prefill_msgs)}"
    prefill_msg = prefill_msgs[0]
    assert prefill_msg.uid == 42
    logger.info(
        f"[rank 1] Received PrefillDoneMsg: uid={prefill_msg.uid}, "
        f"cached_len={prefill_msg.cached_len}, device_len={prefill_msg.device_len}"
    )

    # Receive KV cache via real NCCL transfer (non-blocking recv)
    device_len = prefill_msg.device_len
    config = worker.config
    num_pages = (device_len + config.page_size - 1) // config.page_size
    local_page_indices = worker.scheduler.cache_manager.allocate_for_transfer(num_pages)

    worker.kv_transfer.recv(
        uid=prefill_msg.uid,
        kv_cache=worker.scheduler.engine.kv_cache,
        paged_indices=local_page_indices,
        src_rank=0,
    )

    # Poll until NCCL transfer completes (blocking send on rank 0 + async recv here)
    deadline = time.time() + 30
    status = TransferStatus.BOOTSTRAPPING
    while time.time() < deadline:
        status = worker.kv_transfer.poll(prefill_msg.uid)
        if status in (TransferStatus.SUCCESS, TransferStatus.FAILED):
            break
        time.sleep(0.001)
    assert status == TransferStatus.SUCCESS, f"KV transfer failed, status={status}"
    worker.kv_transfer.cleanup(prefill_msg.uid)
    logger.info("[rank 1] KV cache received via NCCL")

    # Add request and run decode step
    worker.scheduler.add_request(
        uid=prefill_msg.uid,
        input_ids=prefill_msg.input_ids,
        sampling_params=prefill_msg.sampling_params,
        cached_len=prefill_msg.cached_len,
        device_len=device_len,
        local_page_indices=local_page_indices,
    )
    assert worker.scheduler.has_running_reqs()

    batch = worker.scheduler.schedule_batch()
    assert batch is not None and batch.is_decode
    forward_input = worker.scheduler.prepare_batch(batch)
    output = worker.scheduler.forward(forward_input)
    msgs = worker.scheduler.process_batch(forward_input, output)

    assert len(msgs) == 1
    assert msgs[0].uid == 42
    logger.info(f"[rank 1] Decode step produced token: {msgs[0].next_token}")

    done_barrier.wait()


def test_e2e():
    ready_barrier = mp.Barrier(2)
    done_barrier = mp.Barrier(2)

    p0 = mp.Process(target=_run_worker, args=(0, "prefill", ready_barrier, done_barrier))
    p1 = mp.Process(target=_run_worker, args=(1, "decode", ready_barrier, done_barrier))

    p0.start()
    p1.start()

    p0.join(timeout=120)
    p1.join(timeout=120)

    assert p0.exitcode == 0, f"Prefill worker failed with exit code {p0.exitcode}"
    assert p1.exitcode == 0, f"Decode worker failed with exit code {p1.exitcode}"

    logger.info("E2E test PASSED")


if __name__ == "__main__":
    test_e2e()
