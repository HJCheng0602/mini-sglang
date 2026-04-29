from __future__ import annotations

from typing import List
import torch
from minisgl.message import BaseBackendMsg, UserMsg, ExitMsg, AbortBackendMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.message import PrefillDoneMsg
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger

logger = init_logger(__name__)

class PrefillWorkerIO:
    """
    manage the IO for prefill worker
    
    receive UserMsg from tokenizer, send PrefillDoneMsg to decode worker
    """
    def __init__(self, config:PDConfig):
        self.config = config

        self.recv_from_tokenizer : ZmqPullQueue[BaseBackendMsg] = ZmqPullQueue(
            config.zmq_backend_addr,
            create=True,
            decoder=BaseBackendMsg.decoder
        )

        pd_backend_addr = f"ipc:///tmp/minisgl_pd_prefill{config._unique_suffix}"
        self.send_to_decode : ZmqPushQueue[BaseBackendMsg] = ZmqPushQueue(
            pd_backend_addr,
            create=True,
            encoder=BaseBackendMsg.encoder
        )

        logger.info(f"PrefillWorkerIO initialized. recv_from_tokenizer: {config.zmq_backend_addr}, send_to_decode: {pd_backend_addr}")

    def recv_messages(self) -> List[BaseBackendMsg]:
        msgs = []
        while not self.recv_from_tokenizer.empty():
            msg = self.recv_from_tokenizer.get()
            msgs.append(msg)
            # logger.debug(f"PrefillWorkerIO received message: {msg}")
        return msgs
    
    def send_prefill_donw(self, msg: PrefillDoneMsg):
        self.send_to_decode.put(msg)
        logger.debug(f"PrefillWorkerIO sent PrefillDoneMsg: {msg}")
    
    def shutdown(self):
        self.recv_from_tokenizer.stop()
        self.send_to_decode.stop()
        logger.info("PrefillWorkerIO shutdown complete")