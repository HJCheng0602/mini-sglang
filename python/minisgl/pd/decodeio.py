from __future__ import annotations

from typing import List

from minisgl.message import BaseBackendMsg, DetokenizeMsg, BatchTokenizerMsg
from minisgl.pd.config import PDConfig
from minisgl.pd.message import PrefillDoneMsg
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger

logger = init_logger(__name__)

class DecodeWorkerIO:

    def __init__(self, config:PDConfig):

        self.config = config

        pd_backend_addr = f"ipc:///tmp/minisgl_pd_prefill{config._unique_suffix}"
        self.recv_from_prefill : ZmqPullQueue[PrefillDoneMsg] = ZmqPullQueue(
            pd_backend_addr,
            create=False,
            decoder=BaseBackendMsg.decoder
        )

        self.send_to_tokenizer : ZmqPushQueue[DetokenizeMsg] = ZmqPushQueue(
            config.zmq_detokenizer_addr,
            create=True,
            encoder=BaseBackendMsg.encoder
        )

        logger.info(
            f"DecodeWorkerIO initialized. "
            f"recv_from_prefill: {pd_backend_addr}, "
            f"send_to_tokenizer: {config.zmq_detokenizer_addr}"
        )

    def recv_prefill_done(self) -> List[PrefillDoneMsg]:
        msgs = []
        while not self.recv_from_prefill.empty():
            msg = self.recv_from_prefill.get()
            if isinstance(msg, PrefillDoneMsg):
                msgs.append(msg)
        return msgs
    
    def send_detokenize(self, msgs: List[DetokenizeMsg]):
        if not msgs:
            return
        if len(msgs) == 1:
            self.send_to_tokenizer.put(msgs[0])
        else:
            batch_msg = BatchTokenizerMsg(data=msgs)
            self.send_to_tokenizer.put(batch_msg)

    def shutdown(self) -> None:
        self.recv_from_prefill.stop()
        self.send_to_tokenizer.stop()
        logger.info("DecodeWorkerIO shutdown complete")
         
        