from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import SamplingParams
from minisgl.message import BaseBackendMsg
from minisgl.message.utils import deserialize_type, serialize_type

@dataclass
class PrefillDoneMsg(BaseBackendMsg):
    """
    When the prefill is done, the message sent to the decode
    """

    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    cached_len: int
    device_len: int
    kv_cache_indices: torch.Tensor
    src_rank: int                       # source rank used for transfer kv

    def encoder(self) -> dict:
        return serialize_type(self)
    
    @staticmethod
    def decoder(json_dict: dict) -> PrefillDoneMsg:
        return deserialize_type(globals(), json_dict)
    
@dataclass
class KVTransferReq(BaseBackendMsg):

    uid: int
    src_rank: int
    dst_rank: int
    num_pages: int     # number of pages to transfer, 0 means transfer all
    paged_indices: torch.Tensor  # the indices of the pages to transfer, shape (num_pages,)
    chunk_id: int = 0
    is_last_chunk: bool = True

    def encoder(self):
        return serialize_type(self)
    
    @staticmethod
    def decoder(json_dict: dict) -> KVTransferReq:
        return deserialize_type(globals(), json_dict)
    
@dataclass
class KVTransferAck(BaseBackendMsg):

    uid: int
    chunk_id: int
    success: bool
    error_msg: str = ""

    def encoder(self):
        return serialize_type(self)
    
    @staticmethod
    def decoder(json_dict: dict) -> KVTransferAck:
        return deserialize_type(globals(), json_dict)
    

@dataclass
class PrefillWorkerReady(BaseBackendMsg):

    worker_rank: int
    worker_endpoint: str

    def encoder(self):
        return serialize_type(self)
    
    @staticmethod
    def decoder(json_dict: dict) -> PrefillWorkerReady:
        return deserialize_type(globals(), json_dict)
    
@dataclass
class DecodeWorkerReady(BaseBackendMsg):
    
    worker_rank: int
    worker_endpoint: str

    def encoder(self):
        return serialize_type(self)
    
    @staticmethod
    def decoder(json_dict: dict) -> DecodeWorkerReady:
        return deserialize_type(globals(), json_dict)