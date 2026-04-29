from __future__ import annotations

from typing import List, Optional, Set

import torch
from minisgl.core import Batch, Req
from minisgl.message import BaseBackendMsg, ExitMsg, UserMsg, AbortBackendMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.message import PrefillDoneMsg
from minisgl.pd.transfer import KVTransferManager
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.utils import init_logger
from minisgl.engine import ForwardOutput

from .prefillio import PrefillWorkerIO
from .prefillsheduler import PrefillScheduler

logger = init_logger(__name__)

class PrefillWorker:
    def __init__(self, config: PDConfig):
        self.config = config
        self.scheduler = PrefillScheduler(config)
        self.kv_transfer = KVTransferManager(
            config.kv_transfer_backend,
            config.kv_transfer_backend_config
        )

        self.io = PrefillWorkerIO(config)

        self.finished_reqs : Set[int] = set()
        logger.info("Prefill Worker initialized with config: %s", config)
    
    @torch.inference_mode()
    def run_forever(self) -> None:
        logger.info("Prefill worker main loop started")

        while True:
            try:
                self._loop_iteration()
            except KeyboardInterrupt:
                logger.info("Prefill worker received KeyboardInterrupt, shutting down")
                break

        self.shutdown()

    def _loop_iteration(self) -> None:
        msgs = self.io.recv_messages()
        for msg in msgs:
            self._process_message(msg)

        batch = self.scheduler.schedule_batch()

        if batch is not None:
            forward_input = self.scheduler.prepare_batch(batch)
            output = self.scheduler.forward(forward_input)

            self._process_batch(batch, output)

    def _process_message(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, ExitMsg):
            raise KeyboardInterrupt()
        elif isinstance(msg, UserMsg):
            self.scheduler.add_request(msg)
        elif isinstance(msg, AbortBackendMsg):
            self.scheduler.abort_request(msg.uid)
        else:
            logger.warning(f"Prefill worker received unknown message type: {type(msg)}")

    def _process_batch(self, batch: Batch, output: ForwardOutput) -> None:

        next_tokens_cpu = output.next_tokens_cpu
        copy_done = output.copy_done_event
        copy_done.synchronize()

        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                logger.debug(f"Prefill worker skipping transfer for chunked request: {req}")
                continue
            next_token = int(next_tokens_cpu[i].item())

            req.append_host(torch.tensor([next_token], dtype=torch.int32))
            page_indices = self._get_page_indices(req)

            dst_rank = self._select_decode_worker(req)

            self.kv_transfer.send(
                uid=req.uid,
                kv_cache=self.scheduler.engine.kv_cache,
                paged_indices=page_indices,
                dst_rank=dst_rank
            )

            self.io.send_prefill_donw(
                PrefillDoneMsg(
                    uid=req.uid,
                    input_ids=req.input_ids,
                    sampling_params=req.sampling_params,
                    cached_len=req.cached_len,
                    device_len=req.device_len,
                    kv_cache_indices=page_indices.cpu(),
                    src_rank=self.config.tp_info.rank
                )
            )

            logger.info(f"Prefill done for request {req.uid}, sent to decode worker {dst_rank}, next token: {next_token}")

    def _get_page_indices(self, req: Req) -> torch.Tensor:
        table_idx = req.table_idx
        device_len = req.device_len
        page_indices = self.scheduler.engine.page_table[table_idx, :device_len]

        return page_indices
    
    def _select_decode_worker(self, req: Req) -> int:
        # TODO: implement a better scheduling algorithm to select decode worker
        return req.table_idx % self.config.tp_info.size
    
    def shutdown(self) -> None:
        logger.info("Shutting down prefill worker")
        self.kv_transfer.shutdown()
        self.io.shutdown()
        self.scheduler.shutdown()


