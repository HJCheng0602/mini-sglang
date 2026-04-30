from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from minisgl.message import BaseBackendMsg, ExitMsg, AbortBackendMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.message import PrefillDoneMsg
from minisgl.pd.transfer import KVTransferManager, TransferStatus
from minisgl.utils import init_logger
from .decodeio import DecodeWorkerIO
from .decodescheduler import DecodeScheduler

logger = init_logger(__name__)

class DecodeWorker:
    def __init__(self, config: PDConfig):
        self.config = config
        self.scheduler = DecodeScheduler(config)
        self.kv_transfer = KVTransferManager(
            config.kv_transfer_backend,
            config.kv_transfer_backend_config,
        )
        self.io = DecodeWorkerIO(config)
        # uid -> (msg, local_page_indices)
        self.pending_transfers: Dict[int, Tuple[PrefillDoneMsg, torch.Tensor]] = {}
        logger.info("DecodeWorker initialized")

    @torch.inference_mode()
    def run_forever(self) -> None:
        logger.info("Decode worker main loop started")
        while True:
            try:
                self._loop_iteration()
            except KeyboardInterrupt:
                logger.info("Decode worker received KeyboardInterrupt")
                break
        self.shutdown()

    def _loop_iteration(self) -> None:
        # 1. Receive PrefillDoneMsg
        prefill_msgs = self.io.recv_prefill_done()
        for msg in prefill_msgs:
            self._handle_prefill_done(msg)
        # 2. Check pending transfers
        self._check_pending_transfers()
        # 3. Schedule and run decode
        batch = self.scheduler.schedule_batch()
        if batch is not None:
            forward_input = self.scheduler.prepare_batch(batch)
            output = self.scheduler.forward(forward_input)
            detokenize_msgs = self.scheduler.process_batch(forward_input, output)
            self.io.send_detokenize(detokenize_msgs)

    def _handle_prefill_done(self, msg: PrefillDoneMsg) -> None:
        """Initiate KV receive for a completed prefill request."""
        device_len = msg.device_len
        num_pages = (device_len + self.config.page_size - 1) // self.config.page_size
        # Allocate local pages
        local_page_indices = self.scheduler.cache_manager.allocate_for_transfer(num_pages)
        # Start KV receive (non-blocking)
        self.kv_transfer.recv(
            uid=msg.uid,
            kv_cache=self.scheduler.engine.kv_cache,
            paged_indices=local_page_indices,
            src_rank=msg.src_rank,
        )
        self.pending_transfers[msg.uid] = (msg, local_page_indices)
        logger.debug(f"Started KV recv for uid={msg.uid}, num_pages={num_pages}")

    def _check_pending_transfers(self) -> None:
        """Check if KV transfers have completed, then add to scheduler."""
        completed = []
        for uid, (msg, local_page_indices) in self.pending_transfers.items():
            status = self.kv_transfer.poll(uid)
            if status == TransferStatus.SUCCESS:
                self.scheduler.add_request(
                    uid=msg.uid,
                    input_ids=msg.input_ids,
                    sampling_params=msg.sampling_params,
                    cached_len=msg.cached_len,
                    device_len=msg.device_len,
                    local_page_indices=local_page_indices,
                )
                self.kv_transfer.cleanup(uid)
                completed.append(uid)
                logger.info(f"KV transfer completed for uid={uid}")
            elif status == TransferStatus.FAILED:
                logger.error(f"KV transfer failed for uid={uid}")
                self.kv_transfer.cleanup(uid)
                completed.append(uid)
        for uid in completed:
            del self.pending_transfers[uid]
            
    def shutdown(self) -> None:
        logger.info("Shutting down decode worker")
        self.kv_transfer.shutdown()
        self.io.shutdown()
        self.scheduler.shutdown()