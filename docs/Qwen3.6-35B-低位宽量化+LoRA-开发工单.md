# Qwen3.6-35B-A3B 低位宽量化 + LoRA 补偿:开发工单

> 2026-11-04。目标:对 Qwen3.6-35B-A3B 的路由专家做 2bit / 3bit 权重量化,
> 用 LoRA 补偿量化误差。环境:8 卡 × 60 GB 显存。
>
> 本文是**执行工单**:做什么、改哪一行、怎么算做完。
> 每条都有验收判据。**没有验收判据的条目不要动手。**
>
> 同目录的 `DSv41-Flash-LoRA-训练流程与操作.md` 记录的是 DeepSeek-V4.1 的现行做法,
> 本文多处引用它作为对照。参照包在 `/home/ma-user/work/l84414662/qwen36/`。

标记约定:**[已验证]** 是本轮在机器上读代码/算过的;**[假设]** 是还没验证、开工前必须先打正控的。

编号约定:**`§N.M` 只指"阶段 N 的第 M 行工单"**(§D 里那八张表),
这是代码注释和 README 里引用的那套。叙述性章节用字母:§A 速查、§B 前提、
§C 设计决定、§D 阶段表、§E 代码清单、§F 风险登记、§G 开工前确认、§H DSv4.1 参照,
子节如 §B.5、§C.2、§G.1。
(2026-11-06 改。之前顶层章节也用 0..7,于是 `§2.2` 同时是"离线 PTQ 这个设计决定"
和"写 offline_ptq.py 这条工单",`§6.1` 同时是"确认 ViT/vocab"和"加 LoRA trainer 配置"
—— 后一对毫无关系,已经造成过误读。)

---

## A. 结论速查

| 问题 | 答案 |
|---|---|
| 要不要照搬 DSv4.1 那套? | **不要**。它的代码量大部分是它自己的处境逼出来的 |
| 最大的简化 | **不做 int8 冻结基座**。bf16 在 8 卡上装得下 |
| 第二大的简化 | **不寄生推理引擎**。PTQ 按层在单卡上串行做完 |
| 校准能省吗 | **2bit 不能省**。RTN 在 2bit 下没有意义 |
| 量化算法自己写吗 | **不写**。原样搬 `qwen36/quant/core.py`(140 行,无自定义 kernel) |
| 新代码量 | 约 **370 行** |
| 先做哪个位宽 | **3bit 先行**,再加混合精度回退,最后逼 2bit |

### 与 DSv4.1 做法的差异总表

| DSv4.1 的组件 | 本路线 | 为什么能去掉 |
|---|---|---|
| int8 冻结基座(`moe_int8.py` + `unisvq_load.py`) | **不需要** | bf16 70 GB / 8 卡 = 8.75 GB/卡 |
| hook 进 `vllm_ascend` 的 W8A8 FusedMoE method | **不需要** | 离线 PTQ,不需要宿主 |
| `cmd.json` 模式切换 + `drive_ptq.py` HTTP 驱动 | **不需要** | 同上 |
| 16 worker 协调、按 rank 分片 + `assemble_adapter` | **不需要** | 单进程量化,EP=1 |
| `rebuild_from_2bit` 加载时逐层重建 | **不需要** | 离线写出 bf16 checkpoint |
| 服务重启 14 次做累积误差标定 | **不需要** | 按层串行天然具备 |
| engram / MTP 3 层 / `convert_other_bf16.py` | **不需要** | Qwen3.6 没有 |
| tmpfs 腾挪、jemalloc 调参、启动重试 | **不需要** | 产物 7.5 GiB,放持久盘 |
| EP 分片导致 Hessian 只用 1/16 路由量 | **不存在** | EP=1 全量收集 |
| UniSVQ 算法本体(RHT / 码本 / LDLQ) | **原样保留** | 这是真正干活的部分 |
| 混合精度热专家回退(`mp_select.py`) | **重写并提前** | 2bit 下是主路径;输入从寄生 collector 的路由计数换成 `--acts` + router 权重 |
| `target_modules=()`,LoRA 只加路由专家 | **保留** | 否则训出来的和能部署的不是同一个模型 |
| `adapter_save.py` 显式保存 adapter | **保留** | 不存就是白训 |

---

## B. 已核实的前提

### B.1 模型定义已经存在,不用自己写 **[已验证]**

上游 torchtitan 里有 Qwen3.5-35B-A3B 的完整定义,几何和 Qwen3.6-35B-A3B 一致:

`third_party/torchtitan/torchtitan/models/qwen3_5/__init__.py:897` 的 `_35b_a3b()`:

| 项 | 值 |
|---|---|
| `dim` | 2048 |
| `moe_hidden_dim` | 512 |
| `num_experts` | 256 |
| `top_k` | 8 |
| `n_layers` | 40 |
| `n_heads` / `n_kv_heads` | 16 / 2 |
| `head_dim` / `rotary_dim` | 256 / 64 |
| `max_seq_len` | 262144 |
| `vocab_size` | 248320 |

注册名是 `"35B-A3B"`(同文件 line 1094)。
`qwen36` 包里对 Qwen3.6 fixture 跑的形状预检给出的是同一组数(hidden 2048 /
moe_inter 512 / 256 专家 / top-k 8 / 40 层),所以两者对得上。

**但那个 flavor 的注释写的是 "MoE config with vision encoder"** —— 见 §G.1。

### B.2 分组专家 LoRA 工厂是架构通用的 **[已验证]**

这是本路线成立的关键。三件事:

1. **上游 `LoRAConverter` 只认 `Linear`。**
   `third_party/torchtitan/torchtitan/components/lora.py:141`,`convert()` 里
   `isinstance(cfg, Linear.Config)`。MoE 的路由专家是 `[E, F, D]` 的三维参数,
   不是 `nn.Linear`,所以上游那 235 行覆盖不到。这就是 DSv4 的
   `models/deepseek_v4/lora.py` 有 628 行的原因。

2. **路由专家用的是上游通用 `GroupedExperts`。**
   `models/common/config_utils.py:428` 的 `make_routed_experts_config()` 返回
   `RoutedExperts.Config(inner_experts=GroupedExperts.Config(...))`,
   而 `models/common/moe.py:35` 的 `GroupedExperts` 参数名是
   `w1_EFD` / `w2_EDF` / `w3_EFD`。Qwen3.5 和 DeepSeek-V4 走的是同一个。

3. **`grouped_lora_class(parent_cls)` 是以父类为参数的工厂。**
   `torchtitan_npu/models/deepseek_v4/lora.py:177`。它的 `forward` 读的正是上面
   那三个通用参数名,`swiglu_limit` 用 `getattr(self, "swiglu_limit", 0.0)` 取、
   为 0 时跳过 clamp —— **Qwen 没有 swiglu_limit,已经兼容**。
   数学上只是 gate/up/down 加 LoRA delta,没有 DeepSeek 专有的东西。

结论:**这个工厂可以直接用在 Qwen3.6 上。** 绑 DeepSeek 的只有
`DEEPSEEK_V4_LORA_TARGETS`(dense 层目标清单),而我们设 `target_modules=()`,
正好把那部分绕开。

### B.3 显存账:8 卡 × 60 GB **[本轮计算,未实测]**

**训练(LoRA,bf16 基座):**

| 项 | 全量 | 摊到 8 卡 |
|---|---|---|
| bf16 全模型权重 | 70 GB | 8.75 GB/卡 |
| rank16 adapter | 1.85 GB | 0.23 GB/卡 |
| adapter 的 AdamW 状态(fp32 两组) | 7.4 GB | 0.92 GB/卡 |
| adapter 梯度 | 1.85 GB | 0.23 GB/卡 |
| **常驻合计** | | **约 10 GB/卡** |

余约 50 GB 给激活。Qwen3.6 的 hidden 只有 2048(DSv4.1 是 5120),
配 `activation-checkpoint:full` + `local_batch_size 1` 应有大量余量。

**对照:DSv4.1 为什么必须上 int8。** 它的路由专家 543.6 B 参数,bf16 按 16 片摊
是 68.1 GiB/die,超过单 die 61.27 GiB 容量,**模型本身就装不下**。
Qwen3.6 的路由专家只有 32.21 G 参数,是它的 1/16.9。这个约束在本项目不存在。

**离线 PTQ(按层串行,单卡):**

| 项 | 单层占用 |
|---|---|
| 本层路由专家 bf16 | 1.61 GB(805 M 参数) |
| 本层 Hessian @EP=1 | 4.25 GiB |
| 缓存隐状态(512 条 × 2048 tok × 2048 dim × 2 B) | 4.3 GB |
| **合计** | **约 10 GB** |

一张卡装得下,余量极大。Hessian 的数字来自 `qwen36/quant/model_cfg.py`
的 `h_gib_per_layer(1)`,该公式对 DSv4.1 算出 2.82 GiB/层/rank,
对上了 `run_v41.sh` 实测的"3 层 8.5 GB 可以 / 4 层 11.3 GB OOM",所以可信。

**产物尺寸:**

| 位宽 | 产物 |
|---|---|
| 2bit | 7.5 GiB |
| 3bit | 11.3 GiB |

都放持久盘,**不要放 `/run`** —— DSv4.1 风险登记里排第一的就是
"tmpfs 重启全失(已发生两次)"。

### B.4 可复用的量化内核 **[已验证]**

`/home/ma-user/work/l84414662/qwen36/quant/core.py`,140 行,
**纯 torch,无自定义 kernel**。模块文档自述对齐 `lc_qat_torch` 的语义:

| 组件 | 说明 |
|---|---|
| `block_hadamard` / `block_hadamard_inv` | RHT:分块 Hadamard(128)+ 随机符号,输入/输出两侧都做 |
| `LinearCodebook` | 码本 `s * (int_lattice - b) @ G^T`,G 正交(默认 I),s 按格点方差匹配 |
| `block_ldl` | H 的分块 LDL 分解 |
| `ldlq` | 按 d=4 列组从后往前量化,用 LDL 回补未量化列 |
| `unisvq_qdq` | 以上的串联入口 |
| `requant_int8_perchannel` | 逐通道 int8 重量化(本路线暂不用,但留着) |
| `proxy_loss` | `E H E^T` 代理损失,量化质量判据 |

常量:`D_GROUP = 4`、`N_BITS = 2`、`HAD_BLOCK = 128`。
形状整除性:Qwen3.6 的 1024 / 2048 / 512 都被 128 和 4 整除,
**不需要补零、不需要动 Hadamard 阶数**(`qwen36` 的形状预检已验证)。

### B.5 量化范围:只做路由专家 **[已验证]**

DSv4.1 的低位宽量化**只针对路由专家**,Qwen3.6 照此办理。

核实依据:
* `unisvq_runtime.py` 和训练侧的 `unisvq_load.py` 里 **`shared_expert` 一次都没出现**;
* 产物的键是按路由专家编号组织的:`e{gexp}.{w13|w2}.{idx|M|sigma|shape|nbits}`,
  `gexp` 取 0..383,共享专家没有专家号,不在其中;
* `rebuild_from_2bit` 只写 FusedMoE 层的 `layer.w13_weight` / `layer.w2_weight`;
* `config_registry.py` 的注释写明"2bit 的损伤本来就只在路由专家上
  (attention/dense 是 W8A8,没被量化到 2bit)"。

**所以保持原精度的是:** attention(q/k/v/o)、**共享专家**、router 的 gate、
RMSNorm、`tok_embeddings`、`lm_head`。

Qwen3.6-35B-A3B 的参数分布:

| 部分 | 参数量 | 占比 | 处理 |
|---|---:|---:|---|
| 路由专家(256 × 40 × 3 × 2048×512) | **32.21 G** | **92%** | **量化到 2/3 bit** |
| `tok_embeddings`(248320 × 2048) | 0.51 G | 1.5% | 原精度 |
| `lm_head`(同上) | 0.51 G | 1.5% | 原精度 |
| attention(40 层) | 约 0.84 G | 2.4% | 原精度 |
| 共享专家(40 × 3 × 2048×512) | 0.13 G | 0.4% | 原精度 |
| norm 等 | 其余 | — | 原精度 |

三个理由,这三条也是 Qwen3.6 不该扩大范围的理由:

1. **收益几乎全在这 92%。** 剩下 8% 即便也量化到 2bit,省下的不到总量的 8%,
   而 embedding / lm_head / attention 的逐参数敏感度高得多。
2. **LoRA 的范围必须和量化范围一致。** 这是 DSv4.1 付过代价的一条:
   adapter 若加在量化没碰的模块上,那部分 delta 在合并路径里不会被应用,
   **训出来的模型和能部署的模型就不是同一个**(实测 936 个键里 736 个是这类)。
   所以 `target_modules=()` 不是省事,是和量化范围对齐。
3. **混合精度回退也在这个范围内做。** `mp_select.py` 挑的是**热路由专家**,
   让其中约 10% 保持高精度,这是 2bit 下最大的那个杠杆(见阶段 4)。

---

## C. 三个设计决定

和 DSv4.1 不同的地方只有这三处,其余都是照搬或不做。

### C.1 不做 int8 冻结基座,全程 bf16

**DSv4.1 的做法:** 把 2bit 重建出的 W_hat 按输出通道烘成 int8 存进参数,
前向时现场反量化回 bf16 再做 matmul。配套 `moe_int8.py`(property 反量化)
和 `unisvq_load.py`(按 rank 重建)。

**本路线:** 离线把 `dequant(W_nbit)` 直接写成一份 **bf16 的 HF checkpoint**,
训练时当普通权重加载并冻结。

**为什么能这样:** int8 那一层在 DSv4.1 里是**容器**,不是量化方案的一部分。
它存在只因为两件事 —— bf16 装不下(68.1 GiB/die),以及 vLLM-Ascend 上唯一能
寄生的宿主是 W8A8 kernel。Qwen3.6 两条都不成立。

**位宽账不受影响:** 2bit/3bit 是 `W_nbit` 的属性。式子从

```
DSv4.1:  W_deploy = quant_int8( dequant(W_nbit) + (alpha/rank)·B·A )
本路线:  W_eval   =             dequant(W_nbit) + (alpha/rank)·B·A      存 bf16
```

少了最外层,顺带省掉 int8 自己约 0.8% 的相对误差。

**风险与回退:** 如果最终部署形态确定要走 W8A8 容器,那么训练时见到的权重
(bf16 的 W_hat)和部署时的(int8(W_hat))差一层 int8 误差。DSv4.1 自己的文档写
"int8 有足够余量,合并几乎无损",所以这是有依据的近似。
要补回来只需在合并后加一次 `core.py` 的 `requant_int8_perchannel`,
**不影响本工单任何其他步骤**。

### C.2 不寄生推理引擎,离线 PTQ

**DSv4.1 的做法:** 量化发生在 vLLM 服务进程内部。hook 打进
`vllm_ascend/quantization/methods/w8a8_dynamic.py` 的三个源码锚点,
靠 `cmd.json` flag 文件在服务运行中切 calib / quant 模式,
`drive_ptq.py` 通过 HTTP 驱动。

**本路线:** 纯 PyTorch 离线。不改 vllm-ascend 一行。

**为什么能这样:** DSv4.1 之所以寄生,一是 543.6 GB 离线物化不下,
二是要让 Hessian 反映 W8A8 的 GMM 真正吃到的输入(它对 `xe` 做了
`npu_dynamic_quant` 再反量化)。Qwen3.6 的基座是 bf16,**没有这一步**,
直接用 `xe` 即可 —— 这不是简化,是正确性:Hessian 必须反映算子真正吃到的分布。

**连带消掉的:** `install_hook_vllm_ascend.py` 的锚点维护、
启动约一半概率失败的 `HcclAllocComResourceByTiling ret=15`、
`cmd.json` 残留会让新服务在**零 Hessian** 下静默写出垃圾分片
(SNR 看着仍有 9 dB,事后极难发现)。

### C.3 PTQ 按层在单卡串行

只把第 L 层驻留在卡上,喂上一层(已量化)输出的隐状态,
收 Hessian、量化、写产物,再前向产出下一层的输入。标准 GPTQ 式串行。

**白送两个性质:**

1. **累积误差传播。** DSv4.1 为了让上游层带着量化误差标定,要把服务重启 14 次、
   每次调 `UNISVQ_REBUILD_MAX_LAYER=<组首层>`。按层串行天然具备。
2. **EP=1 全量 Hessian。** DSv4.1 的 `collect()` 只看到本 rank 分片内的 token,
   每个专家的 H 只用了真实路由量的 **1/16**,这是已定位的 16× 质量损失,
   在它的方案里被列为最高杠杆的待改项。单进程量化不存在这个问题。
   而且 Qwen3.6 的激活比是 8/256 = 3.1%(DSv4.1 是 6/384 = 1.6%),
   同样 token 数下每专家样本又多一倍。

---

## D. 阶段与工单

### 阶段 0:基线与加载验证(不占卡的部分先做)

| # | 事 | 验收 |
|---|---|---|
| 0.1 | 验证 `LoRAConverter` 能在 Qwen3.5-35B-A3B 的配置树上挂住分组专家 LoRA。**不需要权重、不需要卡**,构造配置树即可 | 转换后的配置树里 `routed_experts.inner_experts` 的 config 类型是 `GroupedLoRAExperts.Config`;打印的 LoRA 行形如 `dense=0 grouped=40` |
| 0.2 | 下 Qwen3.6-35B-A3B 权重,逐文件 Sha256 校验 | 校验全对,条数记录在案 |
| 0.3 | 用真 config 替换 `qwen36/fixtures/qwen36-35b-a3b/config.json` 重跑形状预检 | 退出码 0,"形状预检:全过" |
| 0.4 | 上游 `qwen3_5` 的 `35B-A3B` flavor 做 5 步训练 dry run,HF 格式初始加载 | `exit=0`,loss 有限且非 NaN,**无** "保持随机初始化" 类告警 |
| 0.5 | bf16 基线评测(8 卡 vLLM) | 三数据集出数,作为**分母**记录 |

**0.1 和 0.4 是整条链的闸门**:flavor 加载不了真权重,后面全部排期作废。
先做它们,不要先写量化器。

### 阶段 1:管道脚手架(RTN,一次性)

目的是**一天内把整条回路跑通**,用一个一次性的量化器,不投入 Hessian。

| # | 事 | 验收 |
|---|---|---|
| 1.1 | 写 `fakequant_rtn.py`:遍历 HF checkpoint 的路由专家矩阵,逐组对称 int-N(group 128,沿输入维)量化再反量化,写出新的 bf16 safetensors | 新 checkpoint 能被 0.4 的 dry run 加载;抽查一个矩阵,写出值与内存里的量化结果逐位相同 |
| 1.2 | 对 3bit 产物评测 | 出数。**这个数只用来确认管道通,不作为交付参照** |
| 1.3 | 在它上面跑 20 步 LoRA | loss 下降,adapter 分片落盘 |

**1.1 是脚手架,不是候选交付物。** 2bit 下 RTN 没有意义:DSv4.1 **用了**完整的
RHT + LDLQ + 码本,2bit 裸版 GPQA 还是从 78.28 掉到 48.48。

### 阶段 2:真量化器

| # | 事 | 验收 |
|---|---|---|
| 2.1 | `core.py:12` 的 `N_BITS = 2` 改成读 `UNISVQ_NBITS` 环境变量 | `UNISVQ_NBITS=3` 下 `proxy_loss` 明显低于 2bit |
| 2.2 | 写 `offline_ptq.py`(约 200 行):按层串行,hook MoE block 输入 x 和 router 的 topk_ids,逐专家累 `H13 += xe^T xe` 和 `H2 += a^T a`(a = swiglu(xe @ W13)),调 `core.unisvq_qdq`,写分片 | 单层 dry-run:40 层里挑 1 层跑完,产物键名齐全,每矩阵 SNR 入账 |
| 2.3 | **单层 dry-run 不能省** | SNR 落在合理区间,`proxy_loss` 单调下降 |
| 2.4 | 全量 3bit 量化 | 40 层产物齐,无 `ntok=0` 的 DEGRADED 矩阵 |
| 2.5 | ~~写 `write_dequant_ckpt.py`:产物 → bf16 HF checkpoint~~ **已被 2.2 覆盖**,改为写抽查脚本 `verify/recheck_ptq_shard.py` | 从产物目录读回一个专家矩阵,用同一 `--bits/--damp/--gate-up-order` 重跑 `quantize_layer`,与写出值逐位相同 |

**2.2 的边界修正(2026-11-06,代码写完后)。**

原文写"hook MoE block 输入 x **和** router 的 topk_ids"。实际只需要前者:
**router 就是 checkpoint 里的一个线性层**,拿到 MoE 输入的 hidden states 之后,
topk_ids、每专家的 `xe`、`a = swiglu(xe @ W13)`、两个 Hessian,全部能在 CPU 上
用 checkpoint 自己的权重重算。而且只要 ids 不要权重,所以 router 用的是
softmax 还是 sigmoid、归不归一化都不影响 —— topk 只看 logits 的序。

于是要卡要模型的部分压到只剩**抓各层 MoE 输入的 hidden states**(每层一个
`[N, dim]`),其余全在 `src/offline_ptq.py` 里且可测(55 项自测)。

**另外补一条原文没有的验收点:`--gate-up-order`。**
`a = silu(gate) * up`,融合顺序反了 `a` 整个变样,`H2` 跟着错,w2 的量化也错 ——
**而 SNR 完全看不出来**,因为 SNR 只看 `||W - Ŵ||`,不看 H。这是会静默产出
垃圾的一类,和 §C.2 里点过的 `cmd.json` 残留导致零 Hessian 是同一类失败。
判法:一条校准数据两种顺序各算一次 logits,和 bf16 基座对齐的那个是对的。

**2.5 的变更(2026-11-06)。**
原条目假设 PTQ 先吐一个中间产物格式,再由 `write_dequant_ckpt.py` 转成 HF
checkpoint。实际 `offline_ptq.py run` **直接写出完整的 bf16 HF 目录**:被量化的
分片重写,其余分片和非 safetensors 文件原样 `copy2`(见 `src/offline_ptq.py`
的"写出"段)。没有中间产物,也就没有这一步转换。原验收判据只剩后半句还成立
—— 抽查一个矩阵是否逐位相同,那是个校验脚本,不是 60 行的转换器。

**2.2 的三处和 DSv4.1 的 `collect()` 不同**(照 `l84414662/qwen36/QWEN36_ADAPTATION.md` §3.2):
去掉对 `xe` 和 `a` 的 `npu_dynamic_quant`(bf16 基座没有这一步);
`swiglu_limit` 为 0 时不 clamp(已兼容);
EP=1 所以全局专家号就是 0..255,**加载侧必须用同一个编号方案** ——
这是 `_signs` 确定性的全部依赖。

### 阶段 3:3bit 参照

| # | 事 | 验收 |
|---|---|---|
| 3.1 | 3bit 产物评测,serve 配置与 0.5 完全一致 | 出数。与 0.5 的差就是 3bit 裸版的代价 |

### 阶段 4:混合精度回退

**这不是优化项,是 2bit 的主路径。** DSv4.1 实测:

| 配置 | bit | GPQA |
|---|---:|---:|
| 2bit 裸 | 2.000 | 48.48 |
| 2bit + hot39 回退 | 2.609 | **65.66** |
| 3bit 裸 | 3.000 | 63.13 |

**2.609 bit 带回退比 3.000 bit 均摊更好。** 把 0.39 bit 平摊给全部专家,
不如让 10% 的热专家保持精确。

**但那个 2.609 的口径是 int8。** `model_cfg.py` 的
`bits(K) = N_BITS + (8 - N_BITS)·K/E` —— 保留的专家存成 **int8**,
所以 `2 + 6×39/384 = 2.609`。本路线全程 bf16(§C.1),保留的专家是 16 bit,
同样的保留比例算出来是 `2 + 14×26/256 = 3.422 bit`。**差 0.81 bit,
不是同一个工作点。** 详见下面 4.1/4.2 的修正。

| # | 事 | 验收 |
|---|---|---|
| 4.1 | 写 `src/mp_select.py`:按校准集路由计数挑热专家。**输入不是原版那个 `route_calib_e*.json`**(那是寄生 collector 的产物,本路线没有),而是 `--acts` + checkpoint 自带的 router 权重,用 `offline_ptq.route()` 现算 | 产出 `keep_hot<K>.json`,每层恰好 K 个专家;常量激活夹具下 hot 列表**精确等于**被路由命中的专家号 |
| 4.2 | `offline_ptq.py --keep-hot`:列进来的专家不量化、原权重直通 | 保留的专家矩阵与源**逐位相同**;平均位宽 `NB + (16-NB)·K/E` 记进 `ptq_manifest.json` 和分片 metadata(`ptq_avg_bits`),TAG 照这个数起 |
| 4.3 | 评测 | 出数 |

**4.1/4.2 的修正(2026-11-07,代码写完后)。**

**一、原文的验收公式对,但那个工作点和 DSv4.1 的不可比。**
`NB + (26/256)·(16 - NB)` 是本路线(bf16 保留)的正确算法,结果 **3.422 bit**。
而 DSv4.1 公开的 65.66 是 **2.609 bit**(int8 保留)。三个可比的点:

| 配置 | 保留比 | 平均位宽 | 产物 |
|---|---:|---:|---:|
| DSv4.1 hot39/384 保 int8(**65.66 就是这个点**) | 10.16% | 2.609 bit | — |
| 同保留比例 hot26 保 bf16 | 10.16% | **3.422 bit** | 13.78 GB |
| 同保留比例 hot26 保 int8 | 10.16% | 2.609 bit | 10.51 GB |
| 同位宽预算 hot11 保 bf16 | 4.30% | 2.602 bit | 10.48 GB |

拿 hot26-bf16 的分数去和 65.66 比,是拿高了 0.81 bit 的模型去比,**赢了不说明
方法更好**。要同预算比就用 **hot11**;要同保留比例就得上 int8 容器
(`core.requant_int8_perchannel` 一行,但那是 §C.1 明确不做的事)。
`src/mp_select.py --bits-table` 把三个点都打出来,选哪个是人的决定。

顺带:bf16 保留让 2bit 产物从 8.05 GB 涨到 13.78 GB,**几乎翻倍** ——
这也是 DSv4.1 当初上 int8 容器的原因之一。

**二、原文说"复用 `mp_select.py`,先改拓扑常量",实际是重写。**
原版读 `route41/route_calib_e*.json`,那是寄生在 vLLM 里的 collector 吐的;
本路线没有 collector(§C.2),所以输入换成 `--acts` + checkpoint 的 router 权重,
用和量化时**同一个** `offline_ptq.route()` 现算路由。改动量远超"拓扑常量 ~5 行",
而且那份 clone 是只读的,所以新写在 `src/mp_select.py`。
原版那条"读错 route 目录会按错的拓扑悄悄选出一份看起来正常的 keep 列表"的风险
因此不存在了,换来的新风险是"激活来自错的模型",硬校验改成查 router 形状 /
激活 dim / 专家数三者一致。

**三、`--keep-hot` 的三类静默失败都挡掉了。**
专家号越界(keep 列表来自别的拓扑)、某层保留了全部专家(等于那层没量化)、
值不是整数列表(把 `*_info.json` 当 keep 传)—— 都直接退出。
**要量化的层不在 keep 里**只警告不拦,但会明说"那一层整层走低位宽",
这是原版 `mp_select.py` 列为"缺层"自检的那一条。
另外保留的专家**不记 SNR** —— 未量化矩阵的 SNR 是 +inf,混进统计会把 p50 推到无意义。

**要量的一个变数:** Qwen3.6 的激活比 3.1% 比 DSv4.1 的 1.6% 更分散,
热冷差距可能没那么悬殊,回退的收益需要实测,不要照搬 DSv4.1 的 +17 点。

### 阶段 5:2bit

| # | 事 | 验收 |
|---|---|---|
| 5.1 | 2bit 裸版 + 2bit 混精,各评一次 | 出数 |

**预期比 DSv4.1 掉得更多。** Qwen3.6 的专家矩阵小一个量级
(w13 是 2.10 M 对 23.59 M,w2 是 1.05 M 对 11.80 M),
同样 bits/weight 下可压缩的冗余结构更少。
尤其 `w2` 的输入维只有 512,Hessian 是 512×512,
`_prep_matrix` 的 Tikhonov(`H + 0.01·mean(diag(H))·I`)在小矩阵上影响更大。

### 阶段 6:LoRA

**先在 3bit 上跑通,再上 2bit。** 依据是 DSv4.1 实测的斜率:
位宽是约 +2.6 点 / 0.5 bit,LoRA 是 **+14.8 点条件准确率、0 bit 代价**。
先在误差小的位宽上验证整条链正确,比在 2bit 裸版上拿一个无法归因的数要好。

| # | 事 | 验收 |
|---|---|---|
| 6.1 | 加 35B-A3B 的 LoRA trainer 配置,converter 用 `DeepSeekV4LoRAConverter.Config(rank=16, alpha=32, rank_experts=16, target_modules=())` | 启动日志的 LoRA 行是 `dense=0 grouped=40 rank=16` |

**6.1 的三处修正(2026-11-05 读码核对后)。**

1. **不用改 torchtitan-npu 的文件。** 上游已经有 `qwen35_35b_a3b()`
   (`third_party/.../qwen3_5/config_registry.py:264`),不需要自己写基座配置。
   实现放在本仓库的 `src/qwen36_lora_recipe.py`,它把
   `torchtitan_npu.models.qwen3_5` 的全部名字搬进自己的命名空间,
   所以可以整体替代 `MODULE=`。那个 clone 里有别人未提交的工作,不要动。
2. **converter 不是 `Trainer.Config` 的字段**,是
   `model_registry(flavor, converters=[...])` 的参数
   (`models/qwen3_5/__init__.py:1100`)。拿到 Trainer.Config 再回头挂是挂不上的。
3. **并行度必须改。** 上游 `qwen35_35b_a3b()` 是 `tp=2 / ep=8` = **16 rank**,
   我们只有 8 张卡。改成 `tp=1 / ep=8`,`dp_shard=-1` 推出来是 1,
   256 个专家摊 8 卡 = 32 个/卡。
| 6.2 | 冻结非 adapter 参数:抄 `deepseek_v4_1/parallelize.py:238-249` 那 8 行。判据用"模型里有没有名字含 `lora_` 的参数",**不要用 `getattr(model, "lora_config")`** | 日志 `LoRA: 冻结了 N 个非 adapter 参数`;优化器拿到的参数数 == adapter 张量数 |
| 6.3 | 优化器 AdamW,Muon profile 置 None | 不报 `matched no parameters` |
| 6.4 | 初始权重 = 阶段 2.5 的 bf16 checkpoint,HF 格式、model-only | 日志无"保持随机初始化" |
| 6.5 | `adapter_save.py` 原样复用,设 `ADAPTER_SAVE_DIR` 和 `ADAPTER_SAVE_EVERY` | 日志 `adapter 保存已注册`;落盘分片数 == world_size |
| 6.6 | 语料:先用公开指令数据跑通;要自生成轨迹再用 `gen_trace.py`,**把 DeepSeek 的全角标记换成 Qwen 的 `apply_chat_template`**,并在输出里报告实际用了哪种 | "chat template 成功"必须 100%,回退 0 |

**6.6 的坑有实测代价。** DSv4.1 两个模型目录的 `tokenizer.chat_template`
**根本没设**,脚本一路静默回退成 5 个 ASCII 字面 token,
修好后 LoRA 造成的退化从 +13.1 点压到 +3.5 点。

**修正(2026-11-05):训练这条路本身是会响的,范围比上面写的窄。**
上游 `torchtitan/components/tokenizer.py:89-90` 的 `apply_chat_template`
在没有模板时直接 `raise ValueError("No chat template set.")`,而
`hf_datasets/text_datasets.py:489` 就是调它。所以静默回退只会发生在
**自己写的数据生成脚本**里(DSv4.1 的 `gen_trace.py`),不会发生在
ChatDataLoader 里。`src/prepare_corpus.py` 的作用因此是"第一步就炸,
而不是 8 卡加载完才炸",外加报告模板是从哪个文件读到的 ——
`chat_template.jinja` 优先于 `tokenizer_config.json` 里的键
(tokenizer.py:158-160),两处都有且不同的目录见过。

**rank 怎么选。** adapter 每层参数 = `E × r × (2·dim + 3·moe_inter)`:

| | 每层每 rank 系数 | rank16 全模型 |
|---|---:|---:|
| DSv4.1 | 384 × 17152 = 6.59 M | 4.22 G / 8.4 GB bf16 |
| Qwen3.6 | 256 × 5632 = 1.44 M | **0.92 G / 1.85 GB bf16** |

只有 DSv4.1 的 21.9%,所以 **rank 16 起步**都比 DSv4.1 的 rank 8 更省显存。
上限是**语料**卡的不是显存卡的:单专家 rank16 是 137 K 参数,
要配得上的 token 量按 top-8/256 摊算。

### 阶段 7:合并与复评

| # | 事 | 验收 |
|---|---|---|
| 7.1 | 写 `merge_lora.py`(约 60 行):`W_merged = W_hat + (alpha/rank)·B·A`,写 bf16 checkpoint | `diag/merge_bite.py` 复用:数真正变了多少个码。千分之几说明 merge 是空操作 |
| 7.2 | **`alpha`/`rank` 必须和训练配置一致** | `rank` 从张量形状反推并与 `--rank` 硬校验;`alpha` 必须显式给(无默认值),并记进 `merge_info.json` 和分片 metadata |

**7.2 的验收判据要改,因为情况比原来写的更糟。**
原判据是"从 adapter 文件 metadata 读 alpha 并交叉校验"。读过
`adapter_save.py:54` 之后:它写的 metadata 是
`{"step": ..., "rank": str(_rank())}` —— 那个 `rank` 是**进程 rank**,
不是 LoRA rank,而 **alpha 根本没有被存进 adapter 文件**。所以:

* LoRA rank 可以从 `w13_lora_a` 的 dim 1 反推,这一条能硬校验。
* alpha 无法恢复,只能由人给。`merge_lora.py` 的 `--alpha`
  **刻意不设默认值** —— DSv4.1 的事故正是"有默认值 32.0"造成的。
* 补偿措施:把实际用的 `rank`/`alpha`/`scaling` 写进输出目录的
  `merge_info.json` **和**每个分片的 safetensors metadata,
  让评测端不可能再和训练端分叉。
| 7.3 | 复评,serve 配置与阶段 3/5 完全一致 | 出数,与同位宽裸版对比 |

**7.2 不是形式主义。** DSv4.1 踩过:`assemble_adapter.py` 的 `--alpha` 默认 32.0,
当时没显式传,adapter metadata 记成 32.0,而训练日志是 4.0,
评测又照 metadata 传了 32 —— **实际加上去的 delta 比训练时学到的放大了 8 倍**。
见 §H 的参照表脚注。

---

## E. 代码清单

### 原样复用(不改一行)

| 文件 | 来源 | 作用 |
|---|---|---|
| `core.py` | `qwen36/quant/` | 量化内核。唯一例外是 §2.1 那一行 `N_BITS` |
| `adapter_save.py` | `qwen36/lora/` | 挂 `register_step_post_hook`,框架无关 |
| `diag/merge_bite.py` | `qwen36/lora/diag/` | 数 int8 码变了多少个,比 RMS(dW) 更决定性 |
| `diag/growth_curve.py` | 同上 | ΔW 增长曲线,区分"收敛"和"爬得慢" |
| `grouped_lora_class` | `deepseek_v4/lora.py:177` | 分组专家 LoRA 工厂,架构通用 |

### 要改

| 文件 | 改什么 | 行数 |
|---|---|---|
| `core.py:12` | `N_BITS` 接 `UNISVQ_NBITS` | 1 |
| `qwen3_5/config_registry.py` | 加 35B-A3B + LoRA 的 trainer 配置 | ~40 |
| `qwen3_5/parallelize.py`(或等价挂点) | 冻结非 adapter 参数 | ~10 |
| `gen_trace.py` | 角色标记换 `apply_chat_template` + 报告实际用了哪种 | ~20 |

### 新写

| 文件 | 作用 | 估计 |
|---|---|---|
| `fakequant_rtn.py` | 阶段 1 的一次性脚手架 | ~80 |
| `offline_ptq.py` | 按层串行 Hessian + 量化 + 写分片 | ~200 |
| ~~`write_dequant_ckpt.py`~~ | **不写了**,`offline_ptq.py` 直接出 HF 目录;换成 `verify/recheck_ptq_shard.py` 抽查 | ~40 |
| `merge_lora.py` | 合并 adapter → bf16 checkpoint | ~60 |
| `mp_select.py` | 按路由计数挑热专家 + 位宽账。**不是改原版**,输入换成 `--acts` + router 权重,见阶段 4 的修正 | ~250 |

**新代码合计约 400 行**(含脚手架;不含脚手架约 320)。

### 明确不用

| 文件 | 为什么 |
|---|---|
| `unisvq_runtime.py`(796 行) | 它的价值在 `collect()` 的寄生逻辑和 `rebuild_from_2bit`,本路线都不要。算法部分 `core.py` 已覆盖 |
| `install_hook_vllm_ascend.py` | 不寄生推理引擎 |
| `drive_ptq.py` / `cmd.json` 机制 | 同上 |
| `moe_int8.py` / `unisvq_load.py` | 不做 int8 冻结基座 |
| `assemble_adapter.py` | 单进程量化;adapter 只有 1.85 GB,直接 `full_tensor()` 即可。**除非**训练 world 大到需要分片 |
| `engram_load.py` / `convert_other_bf16.py` | Qwen3.6 没有 engram |
| `deepseek_v4/peft.py` 的 PEFT 导出 / `scripts/lora/merge_adapter.py` | 那是 DSv4 专属,对 V4.1 都不适用,对 Qwen 更不适用 |

---

## F. 风险登记

| 风险 | 性质 | 处置 |
|---|---|---|
| 上游 flavor 加载不了真权重 | **阻塞全部** | §0.4,先做 |
| `w13` 融合顺序猜错 | **静默产生垃圾** | 跑 `verify_recon.py`:同侧 corr ≈ 0.88,交叉配对 corr ≈ 0。先确认 HF checkpoint 存的是融合 `[gate;up]` 还是分开的三个 proj |
| `_signs` 两侧编号不一致 | **静默产生垃圾** | 量化侧和加载侧用同一个全局专家号方案;`verify_rebuild.py` 逐位对 |
| 角色标记静默回退 | 实测代价 +9.6 点 | §6.6 必须报告实际用了哪种 |
| LoRA 加在部署时用不上的模块上 | 训出来的和能部署的不是同一个模型 | `target_modules=()`,验收看 adapter 键是否全含 `routed_experts.inner_experts` |
| adapter 没保存 | 白训几小时 | `ADAPTER_SAVE_DIR` 不设只打 warning,§6.5 的验收必须查落盘 |
| `alpha`/`rank` 不一致 | 数据不可比 | §7.2 交叉校验 |
| 产物放 tmpfs | 重启全失(DSv4.1 已发生两次) | 放持久盘 |
| 多租户 | 长期 | 不动别人的文件夹;清进程按 cgroup + 工号双重设限,**绝不 `pkill -f`** |

---

## G. 开工前必须确认的四件事 **[假设]**

| # | 要确认什么 | 怎么确认 | 不成立的后果 |
|---|---|---|---|
| G.1 | 上游 `_35b_a3b` 注释写 "with vision encoder",vocab 248320。Qwen3.6-35B-A3B 的真实 vocab 和是否带 ViT 要核 | 读真 config.json;看 `qwen3_5/state_dict_adapter.py` 的键映射 | 要么找纯文本 flavor,要么自己加一个 |
| G.2 | `LoRAConverter` 的配置树遍历能否在 Qwen 的层结构下找到 `GroupedExperts.Config` | §0.1,纯 CPU | 要改 converter 的树遍历 |
| G.3 | HF checkpoint 里专家权重是融合 `[gate;up]` 还是分开的 `gate_proj`/`up_proj`/`down_proj` | 读一个分片的键名 | 决定 `offline_ptq` 和 `merge_lora` 的切分逻辑 |
| G.4 | Qwen3.6 的实际推理长度(定 `TRACE_MAXTOK`) | 生成 50 条看 `reasoning_tokens` 的中位和 p75 | 截断比例过高,语料质量塌 |

---

## H. 附:DSv4.1 的实测参照

拿来当靶子,**不是预期值** —— Qwen3.6 的专家矩阵小一个量级,2bit 预期掉得更多。

| 配置 | bit | GPQA | MATH-500 |
|---|---:|---:|---:|
| 2bit 裸 | 2.000 | 48.48 | 92.80 |
| 2bit + hot39 回退 | 2.609 | 65.66 | 94.40 |
| 3bit 裸 | 3.000 | 63.13 | 94.20 |
| 3bit + LoRA | 3.000 | 69.70 | 96.00 |
| W8A8 基线 | 8 | 78.28 | 94.60 |

读这张表的三个要点:

1. **2.609 bit 带回退(65.66)胜过 3.000 bit 均摊(63.13)** —— 混合精度比堆位宽有效。
2. **LoRA 是 +14.8 点条件准确率、0 bit 代价**;位宽斜率只有约 +2.6 点 / 0.5 bit。
   **先做 LoRA 线,不是先堆位宽。**
3. **最后一行的 "3bit + LoRA" 有一处待对账。** 那批结果是在评测侧传
   `UNISVQ_LORA_ALPHA=32`、而训练日志记 `alpha: 4.0` 的情况下跑出来的
   (rank 都是 2)。若该不一致成立,这组数的 scaling 是 16 而非 2。
   **拿它当 Qwen3.6 的靶子之前先确认。**

另两条已证实的结论,直接搬:

* **校准集调参是死路。** v7 加了针对性数据,七个指标全在噪声内,SNR 只 +0.04 dB。
* **语料的杠杆是轨迹长度 + 自己收尾,不是题型覆盖。** mmlu_pro 实测:
  抽不出答案 8.3% → 11.6%,而条件准确率 87.19% → 88.93%(已超基线)。
  **缺口全在收尾,不在内容。**

---

*本工单基于 2026-11-04 对 `l84414662` 工作区的只读调查。
标 [已验证] 的均注明了文件与行号;标 [假设] 的开工前必须先打正控。*
