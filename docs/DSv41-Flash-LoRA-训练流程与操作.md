# DeepSeek-V4.1-Flash LoRA 微调：流程与操作手册

> 调查对象：`/home/ma-user/work/l84414662/torchtitan-npu`（+ 同工作区的外部驱动脚本）
> 调查时间：2026-11-04。调查时 `p5v3_both` 这一版训练**正在跑**（16 卡占用中）。
> 本文只记录**实际存在并已跑通**的东西；推测的部分都显式标了「待验证」。

---

## 0. 结论速查

| 问题 | 答案 |
|---|---|
| 这个仓库支持 DSv4.1-Flash 的 LoRA 微调吗？ | **支持，而且已经跑通**：2026-11-04 完成两版 400 步训练，产出 1.05 GB adapter |
| 是仓库自带的吗？ | **不是**。仓库官方只支持 **DSv4** 的 LoRA；DSv4.1 这条链是本地工作区加的，**尚未提交**（见 §2） |
| 入口配置叫什么 | `deepseek_v4_1_flash_lora_int8_r2`（当前在用）等 5 个，见 §3 |
| 怎么跑一次 | `dsv4/unisvq/dbg_int8_v2.sh`，用环境变量传参，见 §5 |
| 最大的坑 | **4 个环境变量不设会静默训出垃圾**（照样 exit=0、照样出 loss），见 §4 |
| 规模 | 单节点 16 卡，seq 2048，400 步约 2 小时，峰值 51.66 GiB/die（84.31%） |
| 产物 | 每 rank 一个 safetensors 分片 → `assemble_adapter.py` 拼成全局 adapter |
| 和 DSv4 的最大差别 | **DSv4.1 没有 PEFT 导出，也没有离线 merge 工具**，必须走自己的保存+拼装路径 |

---

## 1. 两条不要搞混的 LoRA 路径

仓库里实际有**两套**东西，文档和代码的归属不同，混用会浪费几小时：

### 1.1 DSv4（仓库官方支持，有文档）

* 文档：`docs/feature_guides/deepseek_v4_lora.md`
* 配置：`deepseek_v4_flash_lora`（rank 16 / alpha 32，dense + 专家都加）
* 启动脚本：`examples/deepseek_v4/deepseek_v4_flash_lora_4k_a3.sh`
* **有** PEFT 导出（`adapter_model.safetensors` + `adapter_config.json`）
* **有** 离线合并工具：`python -m scripts.lora.merge_adapter --base-model ... --adapter ... --output ...`
* 检查点管理器：`models/deepseek_v4/peft.py` 的 `PeftCheckpointManager`
  （`periodic_save_adapter_only` / `last_save_in_peft` 这些开关只对 DSv4 有效）

> 已知小坑：`deepseek_v4_flash_lora_4k_a3.sh` 用了 `${SCRIPT_DIR}` 但自己不定义，
> 配 `set -u` 直接失败。调用前要 `export SCRIPT_DIR=examples/deepseek_v4`。

### 1.2 DSv4.1（本地加的，无官方文档，就是本文主题）

* **没有** PEFT 导出。`state_dict_adapter` 把 adapter 键放进了 `_SELF_FILLED`，
  HF 格式保存会**跳过** adapter 键 —— 载入时这是对的，保存时等于丢掉训练结果。
* **没有** 接 `PeftCheckpointManager`（那是 DSv4 专属）。
* 默认 checkpoint manager 存全模型 DCP：35.96 GiB × 16 = **约 575 GB**，不可行。
* 所以保存走自己写的 `adapter_save.py`（只存 `lora_*`，rank2 下共 1.06 GB）。

**结论：DSv4 的文档里关于导出/合并/`checkpoint.*` adapter 开关的那几节，对 V4.1 一律不适用。**

---

## 2. 代码在哪：这条链还没提交

`git status` 显示 DSv4.1 LoRA 相关的改动**全在工作区**，master 上没有：

```
已修改（M）
  torchtitan_npu/models/deepseek_v4/lora.py
  torchtitan_npu/models/deepseek_v4_1/__init__.py
  torchtitan_npu/models/deepseek_v4_1/config_registry.py      <- LoRA 配方在这
  torchtitan_npu/models/deepseek_v4_1/parallelize.py          <- 冻结非 adapter 参数
  torchtitan_npu/models/deepseek_v4_1/sharding.py
  torchtitan_npu/models/deepseek_v4_1/state_dict_adapter.py
  torchtitan_npu/models/deepseek_v4_1/engram/{core,host,lookup}.py
  torchtitan_npu/ops/ascendc/mhc.py
  torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py

未跟踪（??）—— 这四个文件是这条链的核心，仓库里原本没有
  torchtitan_npu/patches/torchtitan/models/common/adapter_save.py    <- 按 rank 存 adapter
  torchtitan_npu/patches/torchtitan/models/common/unisvq_load.py     <- 2bit 产物重建进 int8 基座
  torchtitan_npu/patches/torchtitan/models/common/moe_int8.py        <- int8 冻结基座模块
  torchtitan_npu/patches/torchtitan/models/common/engram_load.py     <- Engram W8A8 表加载
```

另外 `third_party/torchtitan/torchtitan/hf_datasets/text_datasets.py` 也被改了
（加了 3 个 `reason_trace*` 数据集，见 §5.3）。

**操作提示**：`git checkout` / `git stash` / 重新 clone 都会让这条链消失。
动这个仓库前先 `git stash list` 和 `git status` 看一眼。
那些 `*.bak-prelora` / `*.bak-preint8` 文件是作者自己留的回滚点，不要删。

---

## 3. 配置入口：5 个 LoRA 配方

都在 `torchtitan_npu/models/deepseek_v4_1/config_registry.py`，
通过 `MODULE=torchtitan_npu.models.deepseek_v4_1` + `CONFIG=<函数名>` 选。

| 配置名 | 基座 | rank/alpha | indexer 头数 | 说明 |
|---|---|---|---|---|
| `deepseek_v4_1_flash_lora_int8_r2` | int8 冻结 | 2 / 4 | **32（官方发布版）** | **当前在用**。2026-11-04 起整条链基于官方权重 |
| `deepseek_v4_1_flash_lora_int8` | int8 冻结 | 16 / 32 | 32 | 基座约 31.6 GiB/die，单节点 16 片装得下 |
| `deepseek_v4_1_flash_lora_int8_idx64` | int8 冻结 | 16 / 32 | 64 | 只配重打包副本 `/dev/shm/v41_w8a8` 那个 revision |
| `deepseek_v4_1_flash_lora_int8_idx64_r2` | int8 冻结 | 2 / 4 | 64 | 同上的 rank2 版 |
| `deepseek_v4_1_flash_lora` | **BF16** | 16 / 32 | 32 | **单节点装不下**（专家 543.6 B bf16 → 68.1 GiB/die > 61.27）。只能多节点/做对照 |
| `deepseek_v4_1_debugmodel_lora_int8` | int8 | 16 / 32 | — | debug 规模，验管路不验精度 |
| `deepseek_v4_1_debugmodel_lora` | BF16 | 16 / 32 | — | 同上的 bf16 对照组，用来隔离「是 int8 的问题还是 LoRA 的问题」|

### 3.1 选 32 头还是 64 头 —— 选错会直接报错

```
Size mismatch ... layers.14.attn.indexer.weights_proj.weight   32 vs 64
```

* 官方 `DeepSeek-V4.1-Flash-w8a8` 发布版是 **32 头**，
  `dsv4/dsv41-other-bf16` 实测 8 个 indexer 层全是 `(32, 5120)` → 用 `_r2`
* `/dev/shm/v41_w8a8` 是别人把 fp8 原模型转成 W8A8 的产物，indexer 是 **64 头** → 用 `_idx64_r2`
* 历史 adapter `t400a` / `t400b` 训在 64 头那版上，**和现在的 32 头版不可比**

### 3.2 LoRA 只加在路由专家上

`_v41_lora_converters()` 里 `target_modules=()`，即**不加 dense 目标**，只加路由专家。
理由（代码注释原话）：merge 路径（`rebuild_from_2bit` 的 hook）只覆盖路由专家，
attention / router 的 adapter 部署时不会被应用 —— 936 个键里 736 个是它们，
**训出来的和能部署的不是同一个模型**。代价只有 7.2 M 参数（1.3%）。

所以正常日志应该是 `LoRA: dense=0 grouped=40 rank=2`。如果看到 `dense=470` 那是 DSv4 的配置。

### 3.3 int8 基座时 converter 顺序不能反

```python
chain = [Int8FrozenBaseConverter.Config(),      # 必须在前
         DeepSeekV4LoRAConverter.Config(...)]   # 再让 LoRA 包上去
```

`lora.py` 的 `_make_lora_config` 用 `owner = cfg._owner` 取父类，
先换成 int8 的 Config 才会得到 `GroupedLoRAExperts(Int8FrozenGroupedExperts)`。
**顺序反了就是 bf16 基座 + LoRA**，不会报错，只是训错对象。

### 3.4 优化器自动降级为 AdamW

`_v41_optimizer_config()` 检测到配置带 `lora` 字段时：
* `_muon_profile=None` —— Muon 的 pattern 指向的全是基座权重，LoRA 把它们冻了，
  给了会报 `matched no parameters`
* 跳过 Engram 表的 SparseAdam 参数组（表已冻结）

所以命令行要显式 `--optimizer.name AdamW`（§5 的脚本已经带了）。

---

## 4. 五个环境变量：四个不设会**静默**训出垃圾

这是这条链最危险的地方。下面前三个不设，400 步**照样跑完、照样出 loss、照样 `exit=0`**，
只在日志里留一条 warning。

| 变量 | 不设的后果 | 正常值（2026-11-04） |
|---|---|---|
| `UNISVQ_CKPT` | int8 基座**保持随机初始化**（只 warning） | `/run/l84414662/unisvq_ckpt2bit_v41` |
| `ENGRAM_W8A8_DIR` | Engram 表**保持随机初始化**（只 warning） | `/run/l84414662/engram` |
| `ADAPTER_SAVE_DIR` | **adapter 根本不保存，跑完就丢**（只 warning） | `dsv4/adapters/<tag>` |
| `TORCHTITAN_ENGRAM_INT8` | Engram host 表按 **F32** 物化 → 宿主 OOM killer 杀光 | `1` |
| `ADAPTER_SAVE_EVERY` | 默认 100 步存一次 | 长跑设成 `=steps`（只要最后一步） |

### 4.1 `TORCHTITAN_ENGRAM_INT8` 的实测数字

不开：每 rank 两个 22.89 GiB 匿名映射（= 24,000,512 × 256 × 4 B，F32），
16 rank 共 **732 GiB**，天花板 508 GiB → 2026-11-04 四次冒烟全被宿主 OOM killer 杀光。
开了：表 int8 5.72 GiB + scale 0.72 GiB，共 **206 GiB**，省 526 GiB。
而且 `load_engram_int8_` **只认 int8 dtype**，不开它加载器根本不会执行。

### 4.2 所以必须有冒烟门禁

`p5_chain.sh` 的设计：先跑 **5 步**确认那几行载入日志都在，才放 400 步长跑。
门禁的 7 条判据见 §7。**不要跳过这一步** —— 一版 400 步是 2 小时。

---

## 5. 怎么跑一次训练

### 5.1 入口脚本

```
/home/ma-user/work/l84414662/dsv4/unisvq/dbg_int8_v2.sh
```

它**不直接调** `scripts/run_train.sh`，而是委托给仓库自己的 debug 启动器
`examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh`。

**为什么必须绕这一层**（v1 直接调 run_train.sh 的失败教训）：
```
AssertionError: FSDP expects uniform original parameter dtype
                but got {torch.float32, torch.bfloat16}
```
DSv4.1 刻意保留了一批 fp32 参数（`attention.py` 的 `attn_sink`、
`mhc.py` 的 `hc_fn`/`hc_base`/`hc_scale`）。DSv4 没有 mHC 和 attn_sink，
**这就是 DSv4 的 LoRA 能直接跑 bf16 的原因**。
V4.1 需要 `NPU_OPS_OVERRIDES` 里的 `mhc.asc_hc_post` 等覆盖项才成立，
而那些 override 是仓库脚本提供的。

仓库脚本把 `EP=8/DP_SHARD=8` 写死了，但 `"$@"` 排在 `$PARALLELISM_ARGS` 之后，
所以**尾部覆盖能生效**。

### 5.2 环境变量表（`DBG_*` 是 dbg_int8_v2.sh 自己的参数）

| 变量 | 默认 | 400 步长跑用的值 |
|---|---|---|
| `DBG_CONFIG` | `deepseek_v4_1_debugmodel_lora_int8` | `deepseek_v4_1_flash_lora_int8_r2` |
| `DBG_NGPU` | 2 | `16` |
| `DBG_DEVICES` | `0,1` | `0,1,...,15` |
| `DBG_EP` / `DBG_DP` / `DBG_CP` | `=NGPU` / `=NGPU` / 1 | `16` / `16` / `1` |
| `DBG_SEQ` | 512 | `2048` |
| `DBG_STEPS` | 5 | `400` |
| `DBG_TIMEOUT` | 2400 | `7800` |
| `DBG_TAG` | `=DBG_CONFIG` | 自定义，决定日志和产物目录名 |
| `DBG_ASSETS` | `/run/l84414662/dsv41-w8a8` | `/home/ma-user/work/l84414662/dsv41_assets` |
| `DBG_INIT_LOAD` | 空（=随机初始化） | `/home/ma-user/work/l84414662/dsv4/dsv41-other-bf16` |
| `DBG_DATASET` | `c4_test` | `reason_trace_v6` 等 |
| `DBG_PORT` | 6371 | — |
| `DBG_LR` / `DBG_WARMUP` | 可选 / `steps/20+1` | — |

关于 `DBG_ASSETS`：真实 flash 模型带 Engram，要求 token 压缩表形状 `(129280,)`。
仓库的测试 tokenizer 只有 2002 词条，会报
`ValueError: Engram token compression map must have shape (129280,), got (2002,)`。
debug 规模可以用测试 tokenizer，**全量必须给真的**。

关于 `DBG_CP`：`world = DP_REPLICATE × DP_SHARD × CP × TP × PP`。
CP>1 把序列切到多个 rank，是**把 seq 推到 4096 的唯一现成途径**；CP=2 时 DP_SHARD 要减半。

### 5.3 数据集

注册在 `third_party/torchtitan/torchtitan/hf_datasets/text_datasets.py`（本地改动）：

| 名字 | 路径 | 规模 |
|---|---|---|
| `reason_trace` | `dsv4/unisvq/data/trace_ds` | 527 条 / 1.62 M token（旧，留作语料对照） |
| `reason_trace_v6` | `dsv4/unisvq/data/trace_ds_v6` | 1047 条 / 1.18 M token（按 calib_v6 的 User 段重采样） |
| `reason_trace_both` | `dsv4/unisvq/data/trace_ds_both` | 1574 条 / 2.80 M token（两者合并） |

格式与 `c4_test` 一致：逐行 `{"text": ...}`，由 `build_trace_dataset.py` 产出
（已剔掉推理中途被截断的轨迹）。

### 5.4 一条完整的调用（照 p5_chain.sh 的 run_one）

训练在 `dsv4-eval` 容器里跑，脚本在宿主上调：

```bash
docker exec \
  -e DBG_CONFIG=deepseek_v4_1_flash_lora_int8_r2 \
  -e DBG_NGPU=16 -e DBG_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  -e DBG_EP=16 -e DBG_DP=16 -e DBG_CP=1 \
  -e DBG_SEQ=2048 -e DBG_STEPS=400 -e DBG_TIMEOUT=7800 \
  -e DBG_PORT=6371 -e DBG_TAG=myrun \
  -e DBG_ASSETS=/home/ma-user/work/l84414662/dsv41_assets \
  -e DBG_INIT_LOAD=/home/ma-user/work/l84414662/dsv4/dsv41-other-bf16 \
  -e DBG_DATASET=reason_trace_v6 \
  -e UNISVQ_CKPT=/run/l84414662/unisvq_ckpt2bit_v41 \
  -e ENGRAM_W8A8_DIR=/run/l84414662/engram \
  -e TORCHTITAN_ENGRAM_INT8=1 \
  -e ADAPTER_SAVE_DIR=/home/ma-user/work/l84414662/dsv4/adapters/myrun \
  -e ADAPTER_SAVE_EVERY=400 \
  dsv4-eval bash /home/ma-user/work/l84414662/dsv4/unisvq/dbg_int8_v2.sh
```

`DBG_INIT_LOAD` 非空时脚本自动加上（HF 格式、只载模型权重，不载优化器/调度器/数据加载器状态）：
```
--checkpoint.enable --checkpoint.load-only
--checkpoint.initial-load-path <dir>
--checkpoint.initial-load-in-hf --checkpoint.initial-load-model-only
```

脚本自己固定追加的其它参数：`--optimizer.name AdamW`、`--compile.no-enable`、
`--training.local-batch-size 1`、`--training.global-batch-size $NGPU`、
`--training.disable-cuda-graphs`、各并行度、lr-scheduler 的 total/warmup。

### 5.5 环境（容器内）

`dbg_int8_v2.sh` 自己处理好了，列出来是为了排障时知道它动了什么：

| 项 | 值 | 为什么 |
|---|---|---|
| venv | `/dev/shm/l84414662/ttvenv` | torch 2.14.0.dev20260719 + torch_npu 2.14.0.dev20260808 |
| CANN | `/dev/shm/l84414662/cann910/cann/set_env.sh`（9.1.0） | 9.0.1 缺 `aclnnMhcPost` 等符号，融合路径不通 |
| 切换口 | `ASCEND_SET_ENV_PATH` | `run_train.sh` 优先认它，比改 bashrc 干净 |
| torchrun shim | `/tmp/ttbin/torchrun` | `run_train.sh` 直接调 `torchrun`，系统 PATH 上那个属于容器自带的 torch 2.10 |
| PATH 顺序 | `/tmp/ttbin` 必须在 **venv bin 之前** | venv 自带 torchrun，否则 shim 不会被调用 |
| 网络 | 显式给 `Network_Interface=enp23s0f3` / `LOCAL_HOST` / `NODE_IPS` / `NODE_RANK=0` / `NNODES=1` | 容器里没有 `ip` 命令，探网卡那步配 `set -e` 直接死 |
| 行缓冲 | `PYTHONUNBUFFERED=1` | 8 KB 块缓冲下六分钟才刷一次日志，会让存活判据误报卡死（已误杀过健康的跑） |

---

## 6. 实测数据（2026-11-04，p5v1_v6）

| 项 | 值 |
|---|---|
| 配置 | `deepseek_v4_1_flash_lora_int8_r2`，16 卡，EP=16，DP_SHARD=16，CP=1 |
| seq / steps | 2048 / 400 |
| 峰值显存 | **51.66 GiB（84.31%）** |
| 每步 | 17.6 ~ 19.5 s |
| 400 步墙上时间 | 约 2 小时（11:27 → 13:32，含 assemble） |
| tps / tflops / MFU | 105 ~ 117 / 15.1 ~ 16.8 / 4.3 ~ 4.7% |
| loss（step 386→400） | 6.61 → 6.49 |
| grad_norm | 0.13 ~ 0.34 |
| adapter 产物 | 1,053,844,928 B（约 1.05 GB），每 rank 约 66 MB |
| 起跑前 HBM 要求 | 最小空闲 ≥ **56320 MB** |

---

## 7. 门禁：7 条必须出现的日志行

`p5_chain.sh` 的 `gate()`。日志带 ANSI 颜色码，判之前先 `sed 's/\x1b\[[0-9;]*m//g'`。

| # | 判据 | 不过说明 |
|---|---|---|
| 1 | `UniSVQ int8 载入完成: 40 个专家模块` | `UNISVQ_CKPT` 没生效 |
| 2 | `Engram int8 载入完成: 2 张表` | `ENGRAM_W8A8_DIR` 或 `TORCHTITAN_ENGRAM_INT8` 没生效 |
| 3 | `DeepSeek-V4 LoRA: dense=0 grouped=40 rank=2` | converter 配错（看到 dense≠0 就是加到 dense 上了） |
| 4 | **不出现** `保持随机初始化` | 基座/表是随机的，训了白训 |
| 5 | **不出现** `adapter 不会被保存` | `ADAPTER_SAVE_DIR` 没设 |
| 6 | `- step: <steps>` | 没跑到目标步数 |
| 7 | `ls <sdir>/adapter_r*_step<NNNNNN>.safetensors` 正好 **16** 个 | 分片不全，不能 assemble |

判据 6 的写法有讲究：必须锚定 `" - step: +N([^0-9]|$)"`。
裸的 `step: *N` 会误判 —— 每步最后一行是 `elapsed_time_per_step: 14.665s`，含子串 `step: 14`。

---

## 8. adapter 的保存与拼装

### 8.1 保存：各 rank 写自己那片，不做 all-gather

`adapter_save.py` 挂在 `post_optimizer_build_fn` → 优化器的 `register_step_post_hook`，
每 `ADAPTER_SAVE_EVERY` 步存一次，文件名 `adapter_r{rank:02d}_step{step:06d}.safetensors`。

* 只存 `name` 里含 `lora_` 的参数，转 bfloat16 落盘
* 每个 DTensor 的 `global_shape` 和 `placements` 写进 metadata（拼装时按它判，不靠假设）
* **只挂第一个**支持 `register_step_post_hook` 的优化器 —— 挂多个会让计数变成真实步数的倍数
* 不 gather 的理由：gather 要 1 GB 集合通信，且 `full_tensor()` 在冻结参数混杂的模型上更易出意外

### 8.2 拼装

```bash
/dev/shm/l84414662/ttvenv/bin/python -I \
  /home/ma-user/work/l84414662/dsv4/unisvq/assemble_adapter.py \
  --in  /home/ma-user/work/l84414662/dsv4/adapters/<tag> \
  --step 400 \
  --rank 2 --alpha 4 \
  --ep-contiguous \
  --out /home/ma-user/work/l84414662/dsv4/adapters/<tag>_step400.safetensors
```

**`--rank` / `--alpha` 必须跟配置一致**（r2 配的是 alpha=4），
否则写进 metadata 的 scaling 是错的 —— 历史 adapter `t400a`/`t400b` 的 metadata
就记成了 alpha=32.0。

**`--ep-contiguous` 的前提：`EP == world_size`。**
专家 adapter 的实测 placements 是
`(_StridedShard(dim=0, sf=16), Shard(dim=0))`，两个 mesh 轴都切 dim 0，
交错切分按 rank 顺序拼**会错**，所以默认拒绝。
打开它的依据是 `unisvq_load.py` 里已验证的同一个映射
（`ep_rank = mesh.get_local_rank("ep")`，`gbase = ep_rank * 24`）→
rank r 持有全局专家 `[24r, 24r+24)`，连续。
`EP != world_size` 时这个前提不成立，那时应改用保存端的 `full_tensor()`。

输出还带一个 `.inventory.json`（约 19 KB）。

---

## 9. 三版编排：p5_chain.sh

```
/run/l84414662/p5_chain.sh      （宿主上跑，训练那步 docker exec 进 dsv4-eval）
```

四个阶段，每阶段都有不过就停的判据：

| 阶段 | 做什么 | 判据 |
|---|---|---|
| 0 | 等上游量化（P4）收尾，**并等 HBM 真的放掉** | 640 分片齐 + 最小空闲 ≥ 56320 MB |
| 1 | 核产物 | 640 分片 / 40 层 / 每层 16 片 |
| 2 | **冒烟 5 步 —— 门禁在这里** | §7 的 7 条全过，不过就不烧那 5 小时 |
| 3 | 三版 400 步，每版跑完立刻 assemble 到持久盘 | 同上 + assemble 成功 |

三个版本：

| tag | 数据集 | 目的 |
|---|---|---|
| `p5v1_v6` | `reason_trace_v6` | 新语料 |
| `p5v2_old527` | `reason_trace` | 旧语料对照 |
| `p5v3_both` | `reason_trace_both` | 把 token 加回去，判断 rank2 是不是数据受限 |

**三版都是 rank2，不加 rank8。** 理由（实测 token 账）：
旧 527 条 1.62 M token、新 1047 条 1.18 M token，单专家从 25K 掉到 18K；
rank2 单专家参数 34K（1.9:1）、rank8 是 137K（7.6:1）——
数据这一侧更紧了，加 rank 是往过参数化走。

### 9.1 多租户机上的清理约定（照抄这个边界）

脚本有两个清理函数，都**双重设限**：进程必须在指定容器的 cgroup 里，
且 cmdline 里带本人工号。**绝不按进程名全局杀。**

* `cleanup_stale_serve` → 清 `dsv41-official` 里残留的 vllm（阶段 0 用）
* `cleanup_cards` → 清 `dsv4-eval` 里上一版残留的训练进程（每版开跑前用）

来源是 2026-11-04 的一次事故：上游编排器没杀掉自己最后一组标定服务
（16 个 worker 各占 44.8 GB），HBM 最小空闲卡在 17 GB，
门禁等满 20 分钟 die —— **卡白空了 5 小时**。门禁本身没错
（不能拿 17 GB 去起要 51.7 GB/die 的训练），错在只会等、不会清。

### 9.2 判断别人的卡是否空闲：必须连续采样

同一份工作日志里的教训：按**一次** `npu-smi` 快照（AICore 全 0%）判断别人在空转，
实测 60 秒内采 20 次是 **55~87% 持续占用**，8100 端口 1200 条连接，在满负荷跑评测。
20 次里确实有一行全 0 —— **3 秒级空隙真实存在，单点采样不可靠**。

另：等卡脚本第一版有 fail-unsafe bug —— `npu-smi` 查询失败时 awk 没有
"Process memory" 行就输出 0，**查询失败被当成卡空闲**，误触发过一次。
修法：每个 board 输出必须含 `Chip Count`，否则该 board 记 999999 MB（失败即视为忙）。

---

## 10. 文件索引

### 仓库内（`/home/ma-user/work/l84414662/torchtitan-npu`）

| 文件 | 作用 |
|---|---|
| `torchtitan_npu/models/deepseek_v4_1/config_registry.py` | **LoRA 配方入口**（§3），`_v41_lora_converters` / `_wire_unisvq_loader` |
| `torchtitan_npu/models/deepseek_v4/lora.py` | LoRA 模块实现，`DeepSeekV4LoRAConverter`（V4.1 复用它） |
| `torchtitan_npu/models/deepseek_v4_1/parallelize.py` | 并行化末尾冻结所有非 adapter 参数 |
| `torchtitan_npu/patches/.../adapter_save.py` | 按 rank 存 adapter（未跟踪） |
| `torchtitan_npu/patches/.../unisvq_load.py` | 2bit 产物按 rank 重建进 int8 基座（未跟踪） |
| `torchtitan_npu/patches/.../moe_int8.py` | int8 冻结基座模块（未跟踪） |
| `torchtitan_npu/patches/.../engram_load.py` | Engram W8A8 表加载（未跟踪） |
| `third_party/torchtitan/torchtitan/components/lora.py:141` | 上游通用 `LoRAConverter` 基类 |
| `third_party/torchtitan/torchtitan/hf_datasets/text_datasets.py:51-70` | `reason_trace*` 数据集注册（本地改动） |
| `examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh` | 实际被委托的启动器 |
| `docs/feature_guides/deepseek_v4_lora.md` | **DSv4** 的官方 LoRA 文档（V4.1 不适用导出/合并部分） |

### 工作区外部

| 文件 | 作用 |
|---|---|
| `dsv4/unisvq/dbg_int8_v2.sh` | **单次训练入口**（§5） |
| `dsv4/unisvq/assemble_adapter.py` | adapter 拼装（§8.2） |
| `/run/l84414662/p5_chain.sh` | 三版编排 + 门禁（§9） |
| `dsv4/TORCHTITAN_SETUP.md` | 环境安装记录：venv、CANN 9.1、5 个本地补丁、踩过的坑 |
| `dsv4/DSv41-2bit-plan.md` | 整体执行方案（P1~P6），LoRA 是其中的 P5 |
| `dsv4/adapters/` | adapter 产物（`<tag>/` 分片 + `<tag>_step400.safetensors`） |
| `dsv4/logs/dbg2_<tag>.log` | 训练全量日志 |
| `dsv4/logs/p5_outer_<tag>.log` | 外层摘要（配置行 + 关键行 + exit code） |
| `dsv4/logs/p5_chain_report.txt` | 编排报告（含每一条门禁结论） |
| `dsv4/dsv41-other-bf16` | LoRA 训练初始权重（HF，model-only，16.52 GiB，由 `convert_other_bf16.py` 重建） |

---

## 11. 要搬到 Qwen3.6-35B 需要做什么

> 本节是**评估**，不是已验证的流程。

现状（已核实）：

* `torchtitan_npu/models/` 下**没有 qwen3_6 包**，只有 `qwen3_5`
* `qwen3_5` 和 `qwen3_6` 目录里**没有任何 LoRA 相关代码**
  （`examples/qwen3_6/run_qwen3_6_4k.sh` 存在，但模型包不在 torchtitan_npu 里）
* `DeepSeekV4LoRAConverter` 是 **DeepSeek 专属**的：
  `DEEPSEEK_V4_LORA_TARGETS`、分组专家的 w13 融合布局、
  `w1_lora_b`/`w3_lora_b` 共享 `w13_lora_a` 这些都绑 DSv4 的 MoE 结构
* 可复用的通用件是上游基类 `third_party/torchtitan/torchtitan/components/lora.py:141`
  的 `LoRAConverter`

搬运清单（按依赖顺序）：

1. **确认 Qwen3.6-35B 在哪个 module 下**，以及它的 MoE 专家参数布局
   （35B-A3B 是 MoE，分组专家的张量形状和命名决定了 converter 怎么写）
2. 基于上游 `LoRAConverter` 写一个 Qwen 版 converter，或确认 DS 版的
   `grouped_lora_class` 能否直接套上 Qwen 的分组专家类
3. 在 Qwen 的 `config_registry.py` 里加 LoRA 配方，并处理两件 V4.1 踩过的事：
   * 优化器：LoRA 下 Muon profile 要置 None（基座被冻，否则 `matched no parameters`）
   * 并行化末尾冻结非 adapter 参数
4. **保存路径必须先定**：Qwen 侧有没有 PEFT 导出？
   如果没有，`adapter_save.py` + `assemble_adapter.py` 这套可以直接借
   （它们只依赖参数名含 `lora_` 和 DTensor metadata，不依赖 DeepSeek 结构），
   但 `--ep-contiguous` 的前提 `EP == world_size` 要重新核
5. 数据集照 `text_datasets.py` 的三条加（格式就是逐行 `{"text": ...}`）
6. 显存预算重算：DSv4.1 是 int8 冻结基座才在 16 卡装下的；
   35B 规模大概率 bf16 基座就够，不需要 UniSVQ/Engram 那两套加载器
   —— 那也意味着 §4 的四个环境变量坑**基本不适用**，链路会简单得多

最省事的参照顺序：**先读 `docs/feature_guides/deepseek_v4_lora.md`**
（那是仓库官方支持、带 PEFT 导出和离线 merge 的完整形态），
再把本文 §5 的启动方式和 §7 的门禁思路搬过来。
DSv4.1 这条链之所以复杂，主要来自「int8 冻结基座 + Engram 表 + 无 PEFT 导出」这三件，
不是 LoRA 本身复杂。

---

*本文据 2026-11-04 的仓库工作区状态与 `dsv4/logs` 实测日志整理。
`p5v3_both` 当时仍在运行，其结果未纳入。*
