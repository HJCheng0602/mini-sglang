from __future__ import annotations

from typing import List, Optional

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine import Engine
from minisgl.kvcache.base import BaseCacheHandle
from minisgl.message import DetokenizeMsg
from minisgl.scheduler import SchedulerConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.table import TableManager
from minisgl.utils import init_logger, load_tokenizer

logger = init_logger(__name__)

class NullCacheHandle(BaseCacheHandle):
    """A no-op cache handle for requests arriving from prefill worker."""
    def __init__(self, cached_len: int = 0):
        super().__init__(cached_len=cached_len)
    def get_matched_indices(self) -> torch.Tensor:
        raise NotImplementedError("NullCacheHandle has no matched indices")
    

class DecodeScheduler:

    def __init__(self, config: SchedulerConfig):
        self.engine = Engine(config)
        self.device = self.engine.device
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.token_pool = self.table_manager.token_pool
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info("DecodeScheduler initialized")

    def add_request(self, uid: int, input_ids: torch.Tensor,
                    sampling_params: SamplingParams,
                    cached_len: int, device_len: int,
                    local_page_indices: torch.Tensor) -> None:
        """Add a decode request after KV cache has been received."""
        table_idx = self.table_manager.allocate()
        # Set up page table: map table_idx to local page indices
        self.engine.page_table[table_idx, :device_len] = local_page_indices[:device_len]
        # Set up token pool
        self.token_pool[table_idx, :len(input_ids)] = input_ids
        req = Req(
            input_ids=input_ids,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=sampling_params.max_tokens,
            uid=uid,
            sampling_params=sampling_params,
            cache_handle=NullCacheHandle(cached_len),
        )
        self.decode_manager.running_reqs.add(req)
        logger.debug(f"Added decode request uid={uid}, table_idx={table_idx}, "
                     f"cached_len={cached_len}, device_len={device_len}")
        
    def has_running_reqs(self) -> bool:
        return self.decode_manager.runnable
    
    def schedule_batch(self) -> Optional[Batch]:
        if not self.has_running_reqs():
            return None
        return self.decode_manager.schedule_next_batch()
    
    def prepare_batch(self, batch: Batch):
        from minisgl.scheduler.scheduler import (
            _make_positions, _make_input_tuple, _make_write_tuple, ForwardInput,
        )
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
            sample_args=sample_args,
        )
    
    def forward(self, forward_input) -> "ForwardOutput":
        from minisgl.engine import ForwardOutput
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        return forward_output
    
    def process_batch(self, forward_input, output) -> List[DetokenizeMsg]:
        """Post-process: append token, check completion, return DetokenizeMsg list."""
        batch = forward_input[0]
        next_tokens_cpu = output.next_tokens_cpu
        output.copy_done_event.synchronize()
        detokenize_msgs: List[DetokenizeMsg] = []
        finished_reqs: List[Req] = []
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = int(next_tokens_cpu[i].item())
                req.append_host(torch.tensor([next_token], dtype=torch.int32))
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                detokenize_msgs.append(
                    DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished)
                )
                if finished:
                    finished_reqs.append(req)
            for req in finished_reqs:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
        return detokenize_msgs
    
    def abort_request(self, uid: int) -> bool:
        req = self.decode_manager.abort_req(uid)
        if req is not None:
            self._free_req_resources(req)
            return True
        return False
    
    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.engine.shutdown()
        logger.info("DecodeScheduler shutdown")