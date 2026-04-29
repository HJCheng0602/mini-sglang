from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from minisgl.core import Batch
from minisgl.engine import Engine
from minisgl.scheduler import SchedulerConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq

from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput
    from minisgl.message import UserMsg

logger = init_logger(__name__)

class PrefillScheduler:
    def __init__(self, config: SchedulerConfig):
        self.engine = Engine(config)
        self.device = self.engine.device

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)

        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)

        self.prefill_manager = PrefillManager(self.cache_manager, self.table_manager, self.decode_manager)

        self.prefill_budget = config.max_extend_tokens
        self.token_pool = self.table_manager.token_pool

        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)

        torch.cuda.set_stream(self.stream)

        logger.info("Prefill Scheduler initialized with config: %s", config)

    def add_request(self, msg: UserMsg) -> None:
        input_len = len(msg.input_ids)
        max_seq_len = self.engine.max_seq_len
        max_output_len = max_seq_len - input_len

        if max_output_len <= 0:
            logger.warning(f"Input lenth {input_len} exceeds or equals max sequence length {max_seq_len}. Rejecting request.")
            return
    
        if msg.sampling_params.max_tokens > max_output_len:
            msg.sampling_params.max_tokens = max_output_len
        
        self.prefill_manager.add_one_req(msg)

        logger.debug(f"Added request {msg.uid} to prefill manager. Input length: {input_len}, max output length: {max_output_len}, sampling params: {msg.sampling_params}")

    def has_pending_reqs(self) -> bool:
        return self.prefill_manager.runnable
    
    def schedule_batch(self) -> Optional[Batch]:
        if not self.has_pending_reqs():
            return None
        
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        logger.debug(f"Scheduled batch with {len(batch.reqs)} requests for prefill.")
        return batch
    
    def prepare_batch(self, batch:Batch):
        from minisgl.scheduler.scheduler import _make_positions, _make_input_tuple, _make_write_tuple, ForwardInput

        # pad batch for CUDA graph(if needed)
        self.engine.graph_runner.pad_batch(batch)

        self.cache_manager.allocate_paged(batch.reqs)

        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]

        self.engine.attn_backend.prepare_metadata(batch)

        sample_args = self.engine.sampler.prepare(batch)

        return ForwardInput(
            batch=batch,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
            sample_args=sample_args
        )
    
    def forward(self, forward_input) -> ForwardOutput:
        from minisgl.engine import ForwardOutput

        batch, sample_args, input_mapping, output_mapping = forward_input

        batch.input_ids = self.token_pool[input_mapping]

        forward_output = self.engine.forward_batch(batch, sample_args)

        self.token_pool[output_mapping] = forward_output.next_tokens_gpu

        return forward_output
    
    def abort_request(self, uid: int) -> bool:
        seq = self.prefill_manager.abort_req(uid)
        if seq is not None:
            self._free_req_resources(seq)
            return True
        return False
    
    def _free_req_resources(self, req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.engine.shutdown()
        logger.info("PrefillScheduler shutdown")




