# PD分离开发计划 (Prefill-Decode Disaggregation)

## 一、整体架构设计

```
                          +------------------+
                          |    API Server    |
                          +------------------+
                                  |
                    +-------------+-------------+
                    |                           |
           +----------------+          +----------------+
           | Prefill Worker |          | Decode Worker  |
           |  (1个或多个)    |          |  (1个或多个)    |
           |   + Scheduler  |          |   + Scheduler  |
           |   + Engine     |          |   + Engine     |
           +----------------+          +----------------+
                    |                           |
                    +---------------------------+
                    |   KV Cache Transfer Layer |
                    |   (torch.distributed)     |
                    +---------------------------+
```

### 核心思想

PD分离（Prefill-Decode Disaggregation）将LLM推理的两个阶段分离到不同的GPU/机器上执行：

- **Prefill阶段**：计算密集型（Compute-bound），处理完整的prompt
- **Decode阶段**：内存密集型（Memory-bound），逐token生成

分离后可以：
1. 更好地利用GPU资源特性
2. 实现更灵活的资源调度
3. 提高整体吞吐量

---

## 二、核心组件拆分

### 2.1 新增模块结构

```
python/minisgl/
├── pd/                          # PD分离核心模块
│   ├── __init__.py
│   ├── config.py               # PD配置参数
│   ├── transfer.py             # KV Cache传输层
│   ├── prefill_worker.py       # Prefill Worker实现
│   ├── decode_worker.py        # Decode Worker实现
│   ├── router.py               # 请求路由器（可选）
│   └── message.py              # PD专用消息类型
├── scheduler/
│   ├── prefill_scheduler.py    # 新增：专用Prefill调度器
│   └── decode_scheduler.py     # 新增：专用Decode调度器
```

### 2.2 消息类型扩展 (`minisgl/pd/message.py`)

```python
@dataclass
class PrefillDoneMsg(BaseBackendMsg):
    """Prefill完成时发送给Decode Worker的消息"""
    uid: int
    input_ids: torch.Tensor       # 完整的input_ids
    sampling_params: SamplingParams
    cached_len: int               # 已缓存的长度
    device_len: int               # 设备上的长度
    kv_cache_shape: Tuple[int, ...]  # KV Cache的shape信息
    kv_page_indices: torch.Tensor  # KV Cache的page索引

@dataclass
class KVTransferReq(BaseBackendMsg):
    """KV Cache传输请求"""
    uid: int
    src_rank: int
    dst_rank: int
    page_indices: torch.Tensor
    chunk_id: int = 0             # 用于chunked传输
    is_last_chunk: bool = True

@dataclass
class KVTransferAck(BaseBackendMsg):
    """KV Cache传输确认"""
    uid: int
    chunk_id: int
    success: bool
```

---

## 三、KV Cache传输层设计 (`minisgl/pd/transfer.py`)

```python
class KVTransferManager:
    """管理KV Cache在Prefill和Decode Worker之间的传输"""

    def __init__(self, config: PDConfig):
        self.config = config
        self.transfer_stream = torch.cuda.Stream()
        self.pending_transfers: Dict[int, KVTransferState] = {}

    async def send_kv_cache(
        self,
        uid: int,
        kv_cache: MHAKVCache,
        page_indices: torch.Tensor,
        dst_rank: int,
        chunk_size: Optional[int] = None,
    ) -> None:
        """发送KV Cache到Decode Worker"""
        # 使用torch.distributed.isend进行异步发送
        # 支持分块传输
        pass

    async def recv_kv_cache(
        self,
        uid: int,
        kv_cache: MHAKVCache,
        page_indices: torch.Tensor,
        src_rank: int,
    ) -> None:
        """从Prefill Worker接收KV Cache"""
        # 使用torch.distributed.irecv进行异步接收
        pass

    def prepare_transfer_metadata(
        self,
        req: Req,
        kv_cache: MHAKVCache,
    ) -> KVTransferMetadata:
        """准备传输所需的元数据"""
        pass
```

### 传输方式选择

使用 `torch.distributed` 的 `isend`/`irecv` 进行异步传输：
- 支持多种后端（NCCL、Gloo、TCP）
- 支持单机多卡和多机多卡
- 可以与计算重叠

---

## 四、Prefill Worker设计 (`minisgl/pd/prefill_worker.py`)

```python
class PrefillWorker:
    """专用的Prefill Worker"""

    def __init__(self, config: PrefillWorkerConfig):
        self.scheduler = PrefillScheduler(config)
        self.engine = Engine(config)
        self.kv_transfer = KVTransferManager(config)
        self.io = PrefillWorkerIO(config)

    async def run_forever(self) -> None:
        """主循环"""
        while True:
            # 1. 接收用户请求
            msgs = await self.io.receive_requests()

            # 2. 调度prefill批次
            batch = self.scheduler.schedule_prefill_batch(msgs)

            if batch:
                # 3. 执行prefill计算
                output = self.engine.forward_batch(batch)

                # 4. 发送KV Cache到Decode Worker
                for req in batch.reqs:
                    await self.kv_transfer.send_kv_cache(
                        uid=req.uid,
                        kv_cache=self.engine.kv_cache,
                        page_indices=self.get_page_indices(req),
                        dst_rank=self.get_decode_rank(req),
                    )

                # 5. 发送prefill完成消息
                await self.io.send_prefill_done(batch.reqs)
```

---

## 五、Decode Worker设计 (`minisgl/pd/decode_worker.py`)

```python
class DecodeWorker:
    """专用的Decode Worker"""

    def __init__(self, config: DecodeWorkerConfig):
        self.scheduler = DecodeScheduler(config)
        self.engine = Engine(config)
        self.kv_transfer = KVTransferManager(config)
        self.io = DecodeWorkerIO(config)

    async def run_forever(self) -> None:
        """主循环"""
        while True:
            # 1. 接收KV Cache（异步）
            recv_tasks = await self.io.receive_kv_transfers()

            # 2. 接收prefill完成消息
            prefill_msgs = await self.io.receive_prefill_done()

            # 3. 处理接收到的KV Cache
            for msg in prefill_msgs:
                await self.kv_transfer.recv_kv_cache(
                    uid=msg.uid,
                    kv_cache=self.engine.kv_cache,
                    page_indices=msg.kv_page_indices,
                    src_rank=msg.src_rank,
                )
                # 将请求加入decode队列
                self.scheduler.add_decode_req(msg)

            # 4. 调度decode批次
            batch = self.scheduler.schedule_decode_batch()

            if batch:
                # 5. 执行decode计算
                output = self.engine.forward_batch(batch)

                # 6. 发送结果到tokenizer
                await self.io.send_results(batch, output)
```

---

## 六、调度器拆分

### 6.1 PrefillScheduler (`minisgl/scheduler/prefill_scheduler.py`)

```python
class PrefillScheduler:
    """专用的Prefill调度器"""

    def __init__(self, config: SchedulerConfig):
        self.prefill_manager = PrefillManager(...)
        self.table_manager = TableManager(...)
        self.cache_manager = CacheManager(...)

    def schedule_prefill_batch(self, budget: int) -> Optional[Batch]:
        """只调度prefill批次"""
        return self.prefill_manager.schedule_next_batch(budget)
```

### 6.2 DecodeScheduler (`minisgl/scheduler/decode_scheduler.py`)

```python
class DecodeScheduler:
    """专用的Decode调度器"""

    def __init__(self, config: SchedulerConfig):
        self.decode_manager = DecodeManager(...)
        self.table_manager = TableManager(...)
        self.cache_manager = CacheManager(...)

    def add_decode_req(self, msg: PrefillDoneMsg) -> None:
        """添加从Prefill Worker接收的请求"""
        req = Req(
            input_ids=msg.input_ids,
            table_idx=self.table_manager.allocate(),
            cached_len=msg.cached_len,
            device_len=msg.device_len,
            output_len=msg.sampling_params.max_tokens,
            uid=msg.uid,
            sampling_params=msg.sampling_params,
            cache_handle=None,
        )
        self.decode_manager.add_req(req)

    def schedule_decode_batch(self) -> Optional[Batch]:
        """只调度decode批次"""
        return self.decode_manager.schedule_next_batch()
```

---

## 七、启动流程修改 (`minisgl/server/launch.py`)

```python
def launch_pd_server(server_args: PDServerArgs) -> None:
    """启动PD分离的服务器"""
    mp.set_start_method("spawn", force=True)

    # 启动API Server
    api_process = mp.Process(target=run_api_server, args=(server_args,))
    api_process.start()

    # 启动Prefill Workers
    for i in range(server_args.num_prefill_workers):
        p = mp.Process(
            target=run_prefill_worker,
            args=(server_args, i),
        )
        p.start()

    # 启动Decode Workers
    for i in range(server_args.num_decode_workers):
        p = mp.Process(
            target=run_decode_worker,
            args=(server_args, i),
        )
        p.start()

    # 启动Tokenizer/Detokenizer
    # ...
```

---

## 八、配置参数扩展 (`minisgl/pd/config.py`)

```python
@dataclass
class PDConfig:
    """PD分离配置"""
    # 部署模式
    pd_enabled: bool = False
    role: Literal["prefill", "decode", "standalone"] = "standalone"

    # Worker配置
    prefill_worker_addr: str = "localhost"
    prefill_worker_port: int = 29500
    decode_worker_addr: str = "localhost"
    decode_worker_port: int = 29600

    # KV Cache传输配置
    kv_transfer_backend: Literal["nccl", "gloo", "tcp"] = "nccl"
    kv_transfer_chunk_size: int = 0  # 0表示不分块
    kv_transfer_overlap: bool = True  # 传输和计算重叠

    # 路由配置
    router_strategy: Literal["round_robin", "load_balance"] = "round_robin"
```

---

## 九、Chunked Prefill到Decode的传输支持

关键点：
1. Prefill Worker在每个chunk完成后立即开始传输
2. Decode Worker可以接收部分KV Cache并开始准备
3. 使用异步传输和事件通知机制

```python
class ChunkedKVTransfer:
    """支持分块传输的KV Cache传输"""

    async def send_chunked_kv(
        self,
        uid: int,
        kv_cache: MHAKVCache,
        chunks: List[PageChunk],
    ) -> None:
        """分块发送KV Cache"""
        for i, chunk in enumerate(chunks):
            # 发送当前chunk
            await self.send_chunk(uid, kv_cache, chunk)

            # 发送chunk完成通知
            await self.send_chunk_ack(uid, i, is_last=(i == len(chunks)-1))
```

---

## 十、Prefix Cache在Decode节点的支持

Decode节点需要：
1. 维护自己的RadixPrefixCache
2. 接收Prefill节点的cache状态
3. 在decode过程中更新cache

```python
class DecodeCacheManager(CacheManager):
    """支持Prefix Cache的Decode Cache管理器"""

    def import_cache_state(
        self,
        uid: int,
        cache_handle: BaseCacheHandle,
        page_indices: torch.Tensor,
    ) -> None:
        """从Prefill节点导入cache状态"""
        # 将Prefill节点的cache映射到本地
        self.prefix_cache.import_mapping(uid, cache_handle, page_indices)
```

---

## 十一、实施步骤

| 阶段 | 任务 | 预计工作量 |
|------|------|-----------|
| **Phase 1** | 基础框架搭建 | 2-3天 |
| | - 创建PD配置模块 | |
| | - 定义PD专用消息类型 | |
| | - 实现基础KV Transfer Manager | |
| **Phase 2** | Prefill Worker实现 | 3-4天 |
| | - 实现PrefillScheduler | |
| | - 实现PrefillWorker主循环 | |
| | - 实现KV Cache发送逻辑 | |
| **Phase 3** | Decode Worker实现 | 3-4天 |
| | - 实现DecodeScheduler | |
| | - 实现DecodeWorker主循环 | |
| | - 实现KV Cache接收逻辑 | |
| **Phase 4** | 启动流程集成 | 2-3天 |
| | - 修改launch.py支持PD模式 | |
| | - 实现进程间同步机制 | |
| | - 集成测试 | |
| **Phase 5** | Chunked传输支持 | 2-3天 |
| | - 实现分块传输逻辑 | |
| | - 实现传输和计算重叠 | |
| **Phase 6** | Prefix Cache支持 | 2-3天 |
| | - 实现Decode节点的cache导入 | |
| | - 实现cache状态同步 | |
| **Phase 7** | 测试和优化 | 3-5天 |
| | - 单机多卡测试 | |
| | - 多机多卡测试 | |
| | - 性能优化 | |

**总计：约17-25天**

---

## 十二、需要修改的现有文件

1. **`minisgl/server/args.py`** - 添加PD相关参数
2. **`minisgl/server/launch.py`** - 添加PD启动逻辑
3. **`minisgl/scheduler/scheduler.py`** - 拆分调度逻辑
4. **`minisgl/message/backend.py`** - 添加PD消息类型
5. **`minisgl/kvcache/mha_pool.py`** - 添加传输支持方法
6. **`minisgl/core.py`** - 可能需要扩展Req类

---

## 十三、技术风险和注意事项

1. **KV Cache传输性能**：torch.distributed的send/recv在跨机场景下可能不如NCCL直接，需要benchmark
2. **内存管理**：需要确保传输过程中的内存安全，避免use-after-free
3. **错误处理**：网络中断、节点故障等异常情况的处理
4. **CUDA Stream同步**：传输和计算重叠时的stream同步问题
5. **Prefix Cache一致性**：多节点间的cache状态同步

---

## 十四、设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 部署场景 | 单机多卡 + 多机多卡 | 先实现单机多卡，架构预留多机扩展 |
| KV Cache传输 | torch.distributed | PyTorch原生API，支持多种后端 |
| Chunked传输 | 支持 | 降低首次token延迟 |
| Prefix Cache | 支持 | 节省Decode节点显存 |
