# Lora-Quant-Qwen3.6-35B

Qwen3.6-35B-A3B 路由专家的低位宽权重量化(2bit / 3bit)+ LoRA 误差补偿。
目标硬件:8 张 NPU,每张 60 GB 显存。

主文档是 [`docs/Qwen3.6-35B-低位宽量化+LoRA-开发工单.md`](docs/Qwen3.6-35B-低位宽量化%2BLoRA-开发工单.md):
做什么、改哪一行、怎么算做完,每条都有验收判据。

## 快速开始

安装分两个阶段,**因为 conda 和 pip 对 proxy 的要求相反**:
conda 要走 proxy,pip 要关掉 proxy。脚本本身不碰任何 proxy 变量,
由你在两个阶段之间自己开关。

```bash
bash env/1_conda_envs.sh
```

```bash
conda activate lqq-quant && bash env/2_pip_quant.sh
```

阶段 1 只建空的 conda 环境,一个包都不装(**这一步需要 proxy 开着**)。
阶段 2 装包并直接跑两个自测(**这一步需要 proxy 关掉**),
它认当前激活的环境,读 `CONDA_PREFIX`,所以激活哪个就装进哪个。

`env/quant.yml` 刻意**不写 `pip:` 段** —— `conda env create` 会在 conda 操作
内部调 pip,那时 proxy 还开着,两个要求没法在同一个进程里同时满足。
量化线的 pip 依赖单独放在 `env/quant-requirements.txt`。

手动装也完全可以,阶段 2 等价于:

```bash
pip install -U pip setuptools wheel && pip install -r env/quant-requirements.txt && python -I verify/test_core_cpu.py && python -I verify/test_fakequant_cpu.py
```

量化线只要 torch + safetensors,**不需要 NPU、不需要 torchtitan**。
两个自测都不碰真权重、不占卡,**先让它们过再往下走**。

拿到真权重后,第一步是看键名分类对不对(只读 header,不写文件):

```bash
python -I src/fakequant_rtn.py --in /path/to/Qwen3.6-35B-A3B --scan
```

确认分类无歧义后再做 3bit 假量化:

```bash
python -I src/fakequant_rtn.py --in /path/to/Qwen3.6-35B-A3B --out /path/to/qwen36-3bit-rtn --bits 3 --group 128
```

**先看磁盘。** 实测这个 checkpoint 的 26 个分片**每一个都含路由专家**,
所以全部要重写,产物是一份完整的约 70 GB 副本,不是增量。
输出目录必须不存在或为空,脚本拒绝覆盖。

产出的是一份 **bf16 的 checkpoint**(量化再反量化),所以任何普通 bf16 推理或
训练路径都能直接加载,不需要低位宽 kernel。位宽是 Ŵ 的信息量属性,不是容器属性。

## 量化范围:只做路由专家

和 DeepSeek-V4.1 的现行做法一致(已核实:量化 runtime 与训练侧加载器里
`shared_expert` 一次都没出现,产物按路由专家编号组织,重建只写 FusedMoE 的
`w13_weight` / `w2_weight`)。

下表是 2026-11-05 对官方权重 `--scan` 的**实测**结果:

| 部分 | 张量数 | 参数量 | 占比 | 处理 |
|---|---:|---:|---:|---|
| 路由专家(融合布局) | 80 | **32.212 G** | **89.6%** | **量化到 2/3 bit** |
| Gated Delta Net(线性注意力) | 240 | 1.012 G | 2.8% | 原精度 |
| 其余(ViT / embedding / lm_head / 全注意力 / norm / 共享专家) | 725 | 2.728 G | 7.6% | 原精度 |

实测确认的三件事:

* **专家是融合布局**,外面还套了一层多模态前缀:
  `model.language_model.layers.{i}.mlp.experts.gate_up_proj` 是 `[256, 1024, 2048]`,
  `down_proj` 是 `[256, 2048, 512]`。80 个张量 = 40 层 × 2,**40 层全是 MoE**。
  参数量对得上解析式 `40 × 256 × (1024×2048 + 2048×512) = 32.212 G`。
  融合顺序(`[gate;up]` 还是 `[up;gate]`)对假量化**不影响** —— qdq 按输出行
  独立做,每行自己一套 scale。顺序只在 LoRA 合并时才要紧。
* **是混合注意力架构**:40 层里 30 层是 Gated Delta Net(`linear_attn.*`,
  带 `A_log` / `dt_bias` / `conv1d` 这些 SSM 参数),只有 10 层是全注意力。
  线性注意力也不量化 —— 它是注意力不是专家,而且 `conv1d.weight` 是
  `[8192, 1, 4]`,每通道只 4 个元素,`group=128` 除不开。
* **带 ViT**(`model.visual.*`),本阶段不量化。

`fakequant_rtn.py` 的范围判定**不靠假设**:先 `--scan` 打出分类供人工核对,
任何"看着像权重但没被分类"的键都会让脚本报错退出。静默跳过会产出一个
看着正常、实际只量化了一部分的 checkpoint。

**LoRA 的范围必须和量化范围一致**(`target_modules=()`)。这是 DeepSeek-V4.1
付过代价的一条:adapter 若加在量化没碰的模块上,那部分 delta 在合并路径里
不会被应用,训出来的模型和能部署的就不是同一个。详见工单 §1.5。

## 目录

```
docs/
  Qwen3.6-35B-低位宽量化+LoRA-开发工单.md   主文档,带验收判据
  DSv41-Flash-LoRA-训练流程与操作.md        对照:DeepSeek-V4.1 的现行做法
env/
  1_conda_envs.sh          阶段 1:只建 conda 环境(这一步要 proxy 开)
  2_pip_quant.sh           阶段 2:量化线装包 + 自测(这一步要 proxy 关)
  2_pip_train.sh           阶段 2:训练线装包 + 四道闸门(这一步要 proxy 关)
  quant.yml                量化线的 conda 环境,刻意不含 pip 段
  quant-requirements.txt   量化线的 pip 依赖
src/
  unisvq_core.py           量化内核:RHT / 码本 / LDLQ / RTN / Hessian 阻尼 / int8 容器
  fakequant_rtn.py         RTN 假量化,HF checkpoint -> bf16 checkpoint
  offline_ptq.py           工单 §2.2:离线 PTQ(RHT + 码本 + LDLQ),逐专家 Hessian
  qwen36_lora_recipe.py    工单 §6:LoRA trainer 配置(可整体替代 MODULE)
  prepare_corpus.py        工单 §6.6:语料归一化 + chat template 前置验死
  merge_lora.py            工单 §7.1/§7.2:adapter 合并回 bf16 checkpoint
verify/
  test_core_cpu.py         内核自测,24 项,纯 CPU
  test_fakequant_cpu.py    端到端自测,合成 checkpoint
  test_offline_ptq_cpu.py  PTQ 自测,55 项,含 2bit 三档对比
  test_merge_lora_cpu.py   合并自测,22 项,纯 CPU,参考值用 einsum 独立算
  probe_lora_tree.py       工单 0.1:LoRA 能否挂上 Qwen 的配置树(需训练环境)
```

`src/unisvq_core.py` 是内部 UniSVQ 参考实现的 vendoring 副本,相对原版只改了
位宽接线(`UNISVQ_NBITS`,2–8 可配;原版把 2 写死成模块常量)并补了几个辅助函数,
改动在文件头有记录。

## 进度

| 工单项 | 状态 |
|---|---|
| 开发工单 | **完成** |
| §1.5 量化范围确认 | **完成**,只做路由专家 |
| §2.1 `N_BITS` 接环境变量 | **完成** |
| 内核 vendoring + 自测 | **完成**,24 项全过 |
| §1.1 `fakequant_rtn.py` | **完成**,端到端自测全过 |
| §0.1 LoRA 配置树检查 | 脚本就绪,**待在训练环境执行** |
| §0.2 下权重 / §0.3 形状预检 | 未做 |
| §2.2 `offline_ptq.py` | **代码完成**,55 项自测全过;真激活未跑过 |
| §2.3 单层 dry-run | 合成数据上过了;**真权重 + 真激活未做** |
| §6.1–6.5 LoRA trainer 配置 | **代码完成**(`qwen36_lora_recipe.py`),**未上卡跑过** |
| §6.6 语料 + chat template | **完成**,两条路径都冒烟过 |
| §7.1/§7.2 `merge_lora.py` | **完成**,22 项自测全过 |
| §6 LoRA 训练本身 | 未做,卡被占 + 训练环境未建 |

### LoRA 线的三件代码就位之后,还差什么

1. **训练环境**(`env/2_pip_train.sh`,要 proxy 关)—— 四道闸门一次都没过。
2. **§0.1 闸门**:`verify/probe_lora_tree.py`,纯 CPU 不占卡,但需要上面那个环境。
3. **一张空闲卡**跑 `qwen36_35b_a3b_lora_smoke`(20 步)。

第 3 步之前,`qwen36_lora_recipe.py` 的全部论证都只是静态读码得来的 ——
见该文件头部"为什么能复用 DSv4 的 LoRA converter"。静态部分已逐条核对到
行号,但 `convert()` 走完之后配置树长什么样,只有 §0.1 说得清。

## 实测结果

### 量化内核(`test_core_cpu.py`)

在 Qwen3.6-35B-A3B 的**真实矩阵形状**上(w13 `[1024, 2048]`、w2 `[2048, 512]`):

| 矩阵 | bits | SNR | LDLQ 代理损失 | RTN 代理损失 | LDLQ 增益 | 单矩阵耗时(CPU) |
|---|---:|---:|---:|---:|---:|---:|
| w13 | 2 | 8.96 dB | 1.192e5 | 1.235e5 | 1.04x | 2.2 s |
| w13 | 3 | 12.35 dB | 5.424e4 | 5.633e4 | 1.04x | 2.2 s |
| w2 | 2 | 8.67 dB | 1.369e4 | 1.552e4 | 1.13x | 0.6 s |
| w2 | 3 | 12.04 dB | 6221 | 7102 | 1.14x | 0.6 s |

另:int8 容器相对误差 0.74%(对上内部实测的约 0.8%);
分组 RTN(group 128)2bit 2.31 dB / 3bit 11.26 dB / 4bit 18.63 dB。

**读这张表必须带上的前提:权重是合成高斯、Hessian 是合成的(各向异性约 31x)。**
所以这些是**管路正确性的 sanity check,不是精度预测**。真实激活 Hessian 的
各向异性远强于此,而 LDLQ 的优势随各向异性增长 —— 表里 1.04 到 1.14 倍的增益
**是下界,不能用来论证 LDLQ 不值得**。

### 五个会改排期的发现

**一、RTN 在 3bit 可当脚手架,在 2bit 必须放弃。**

| bits | RTN(group 128) | LDLQ + RHT | 差距 |
|---:|---:|---:|---:|
| 2 | 2.31 dB | 8.67 dB | **6.4 dB** |
| 3 | 11.26 dB | 12.04 dB | 0.8 dB |

3bit 下只差 0.8 dB,**完全够打通整条回路**;2bit 下差 6.4 dB,
所以 2bit 必须等 Hessian 路线(`offline_ptq.py`),不能用 RTN 交付。
`fakequant_rtn.py` 在 `--bits 2` 时会打这条警告。

**二、专家权重朝向:实测布局和 `state_dict_adapter` 期望的正好转置。**

`third_party/torchtitan/torchtitan/models/qwen3_5/state_dict_adapter.py:272-284`
期望 HF 的 `gate_up_proj` 是 `[E, dim, 2*hidden]` = `[256, 2048, 1024]`,
在**最后一维**融合,`chunk(2, dim=-1)` 之后转置成 TT 的 `w1_EFD`。
但 2026-11-05 对官方权重扫描记下的是 `[256, 1024, 2048]` = `[E, 2*hidden, dim]`,
在**第 1 维**融合。`down_proj` 同理(实测 `[256, 2048, 512]`,adapter 期望
`[256, 512, 2048]`)。

两者只可能有一个对。后果:

* **工单 §0.4(用上游 flavor 加载真权重)会在 `w1_EFD` 上报 shape mismatch
  并直接失败** —— 这是响的,不是静默的。真发生了,改的是 `state_dict_adapter`,
  而且那是上游文件,要走 patch 而不是直接改 clone。
* `merge_lora.py` 因此**按实际形状判朝向**,不假设。`dim=2048 != 2*hidden=1024`
  所以判得出来;两者相等时它会拒绝并要求 `--gate-up-layout` 显式指定,
  自测里有这一项。
* 量化那一步不受影响:`quant_tensor` 按输出行独立做 qdq,每行自己一套 scale,
  转置只改"哪一维是组",group 128 两种朝向都整除。

**三、2bit 有一条不需要任何校准数据的可用路线,所以它不再卡在"抓激活"上。**

`offline_ptq.py` 的 `--hessian identity` 下(合成矩阵,`test_offline_ptq_cpu.py` C 段):

| bits | 方法 | SNR | proxy_loss |
|---:|---|---:|---:|
| 2 | RTN group128 | 2.35 dB | 6.83e2 |
| 2 | RHT + 码本(H=I) | **9.11 dB** | **1.45e2** |
| 2 | RHT + LDLQ(H=I) | 9.11 dB | 1.45e2 |
| 2 | RHT + LDLQ(真 H) | 8.29 dB | **1.17e2** |

两件事:

* **H=I 时 LDLQ 和"只有码本"的结果逐位相同。** 数学上就该如此 —— 没有非对角
  结构,误差反馈没有方向可推。所以 `--hessian identity` 买到的是
  **RHT + 码本**,不是 LDLQ。
* 但它相对 RTN 已经是 **proxy_loss 降到 21%、SNR +6.8 dB**。
  也就是说 2bit 的下限不是 RTN 那个 2.35 dB,而是 9 dB 量级,
  **不需要一个 token 的校准数据**。抓激活那条链(要卡、要跑前向)
  因此从 2bit 的**阻塞项**降级成**优化项**。

**四、一旦用上真 Hessian,SNR 会下降 —— 那是对的,不是回退。**

上表 2bit:SNR 从 9.11 掉到 8.29 dB,而 proxy_loss 从 1.45e2 降到 1.17e2。
Hessian 感知量化干的就是这件事:把误差挪到激活不激发的方向上去,
代价是 `||W - Ŵ||` 变大。

后果是**判据必须换**:跨方法比较时看 `proxy_loss`,不要看 SNR。
只盯 SNR 的人会把真 Hessian 那一版读成回退,然后往反方向"修"。
这也是为什么 `--gate-up-order` 弄反特别危险 —— SNR 根本看不见 H 错没错。
`ptq_manifest.json` 两个都记,比较时用后者。

**五、纯 CPU 做全量 PTQ 约 8 小时,不是工单里估的 2–4 小时。**
按实测单矩阵耗时,每层 256 专家约 12 分钟,40 层约 8 小时(单线程)。
工单那个估计隐含了用 NPU 做 `ldlq` 里的矩阵乘,而那个 Python 循环
(n=2048 时 512 次迭代)不随设备变快,所以上卡也压不到 1 小时以下。
**排期按卡的可用性分两档写。**

## 下一步

与卡的可用性解耦,按这个顺序:

1. ~~下权重、核对键名分类、3bit 假量化~~ —— **已完成**,7.1 分钟,
   SNR min 10.70 / p50 11.12 / max 11.68 dB。
2. **不需要卡**:建训练环境(`env/2_pip_train.sh`,proxy 要关),
   跑 `verify/probe_lora_tree.py`(§0.1)。**这是整条 LoRA 线的闸门。**
3. **不需要卡**:`prepare_corpus.py --tokenizer <tokenizer目录> --synthetic 64`
   先把 chat template 验死,顺手产出冒烟语料。
4. 需要一张卡:`qwen36_35b_a3b_lora_smoke`(20 步)。看三件事 ——
   日志里 `dense=0 grouped=40`、`冻结了 N 个非 adapter 参数`、adapter 分片落盘。
5. 需要 8 张卡:bf16 基线评测 + 3bit 裸版评测(复用
   `eval_qwen3.6-35b_vllm`,`HIF4_MODE=off` + `MODEL_PATH` 覆盖,**两次必须给
   不同的 `TAG`**,否则第二次会 resume 第一次的生成结果)。
6. 需要 8 张卡:`qwen36_35b_a3b_lora_r16` 正式训练 → `merge_lora.py` → 复评。
7. 2bit:`offline_ptq.py --hessian identity`,**不需要校准数据**,先出一版
   (发现三);要上真 Hessian 再补抓激活那一步。RTN 在 2bit 不可用。

## 离线 PTQ 怎么跑

```bash
python src/offline_ptq.py run --in <ckpt> --out <dst> --bits 2 --hessian identity
```

要用真 Hessian 的话,`--acts` 指向一个目录,每层一个 `layer<NN>.pt`,
内容是该层 **MoE 输入** 的 hidden states,形状 `[N, dim]`:

```bash
python src/offline_ptq.py run --in <ckpt> --out <dst> --bits 2 --acts acts/ --layers 20
```

**抓 hidden states 的那段 hook 不在本仓库**,因为它依赖用哪个框架跑前向,
而训练环境还没建起来,写了也验不了。除它之外的全部都在:router 本身就是
checkpoint 里的一个线性层,所以 topk_ids、每专家的 `xe`、
`a = silu(gate)*up`、两个 Hessian,都由 `offline_ptq.py` 在 CPU 上重算。

两个必须显式确认的开关:

| 开关 | 判错的后果 |
|---|---|
| `--layout` | 形状能判出来(2048 != 1024),撞车时拒绝 |
| `--gate-up-order` | **会静默出错**:顺序反了 `a` 整个变样 → `H2` 错 → w2 量化错,而 SNR 看不出来 |

`--gate-up-order` 的判法:拿一条校准数据,两种顺序各算一次 logits,
和 bf16 基座对齐的那个是对的;或者直接看官方 modeling 代码。
实际用的顺序记在 `ptq_manifest.json` 里。

## LoRA 训练怎么起

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python src/prepare_corpus.py --tokenizer <tokenizer目录> --synthetic 64 --out corpus/smoke.jsonl
```

```bash
cd ~/src/torchtitan-npu && NGPU=1 MODULE=qwen36_lora_recipe CONFIG=qwen36_35b_a3b_lora_smoke HF_ASSETS_PATH=<tokenizer目录> INITIAL_LOAD_PATH=<假量化ckpt> DATA_FILES=<corpus/smoke.jsonl 的绝对路径> ADAPTER_SAVE_DIR=<adapter输出目录> ADAPTER_SAVE_EVERY=20 bash scripts/run_train.sh
```

`MODULE=qwen36_lora_recipe` 不是 `torchtitan_npu.models.qwen3_5` —— 那个模块的
全部名字都被搬进了我们这个模块,所以它是**超集**,而 torchtitan-npu 的
clone 一个文件都不用改(那里有别人未提交的工作)。

训完合并:

```bash
python src/merge_lora.py --base <假量化ckpt> --adapter <adapter输出目录> --rank 16 --alpha 32 --dry-run
```

`--alpha` **故意没有默认值**:`adapter_save.py` 写的 metadata 里那个
`rank` 是**进程 rank**,alpha 根本没存进去。DSv4.1 就是在这里把 delta
放大了 8 倍(训练日志 alpha=4,评测照默认值传了 32),那批 `3bit+LoRA`
参照数因此全部不可用。合并后的 `merge_info.json` 和分片 metadata 里会
记下实际用的 scaling,评测端照它核对。

## 训练环境安装

工单阶段 6(LoRA 训练)才需要。量化线用不到。同样分两个阶段,
proxy 的开关同上由你控制:

```bash
bash env/1_conda_envs.sh
```

```bash
conda activate lqq-train && bash env/2_pip_train.sh
```

阶段 1 一次把量化线和训练线两个环境都建好(只要量化线就 `WANT_TRAIN=0`)。
阶段 2 装包并过四道闸门,每道失败都会打出下一步该查什么:

| 闸门 | 验什么 | 典型失败 |
|---|---|---|
| 1 | torch 装上且能 import | 直链被回收 |
| 2 | **torch_npu 能 import** | ABI 不符 / CANN 找不到 |
| 3 | torchtitan_npu 能 import | torch 太旧(缺 `DataParallelMeshDims`) |
| 4 | 分组专家 LoRA 挂上 Qwen3.5-35B-A3B | 这是 LoRA 线的闸门 |

全程不占 NPU(闸门 4 置空 `ASCEND_RT_VISIBLE_DEVICES`,纯 CPU 跑配置树)。
真正上卡的 5 步 dry run 是脚本最后单独给出的一条命令,留给卡空闲时跑。

版本组合(2026-11-05 复查,两个 wheel 都仍可下载):

| 包 | 版本 |
|---|---|
| torch | `2.14.0.dev20260719+cpu` cp312 aarch64(直链) |
| torch_npu | `2.14.0.dev20260808` |
| triton-ascend | `3.2.1` |
| spmd_types | `0.2.3`(不是 0.3.0,torchtitan 0.3.0 钉的是这个) |

**torch 和 torch_npu 必须成对换,不要只升一个。** `aten::_scaled_grouped_mm_v2`
的 `Tensor[]` 参数在 torch nightly 0719 到 0731 之间从 `c10::ArrayRef` 改成了
`c10::IListRef`,而 `torch_npu dev20260808` 是照 0719 编的 —— 跨过去会在
`import torch_npu` 时报 `Mismatch in kernel C++ signatures` 直接 abort。
索引里另有 `torch_npu 2.14.0.dev20260918`,换它就得同步换 torch 的日期。

## 已知的环境陷阱

这些来自内部的安装记录,换机器前值得先看一眼:

* **torch 版本被钉死。** torchtitan-npu 要 `torch==2.14.0.dev20260719`,
  用更新的 nightly 会在 `import torch_npu` 时因 ABI 不符直接 abort。
  量化线不受影响(2.7 起都可以)。
* **环境不要建在 `/dev/shm`。** 上一份训练环境就建在那里,随重启整个消失。
* **CANN 版本决定融合算子可用性。** 9.0.1 缺 `aclnnMhcPost` / `aclnnSparseFlashMla`
  等符号,9.1.0 才有;而且 `libopapi.so` 在 ops 包里,只装 toolkit 验不出来。
* **产物不要放 tmpfs。** 2bit 产物 7.5 GiB、3bit 11.3 GiB,放持久盘。
