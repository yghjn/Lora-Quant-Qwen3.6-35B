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
不会被应用,训出来的模型和能部署的就不是同一个。详见工单 §B.5。

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
  qwen36_lora_compat.py    converter 兼容层:已提交的 lora.py 把根配置写死成 DSv4,Qwen 过不去
  qwen36_adapter_save.py   adapter 周期落盘(原先依赖的文件在干净 clone 里不存在)
  prepare_corpus.py        工单 §6.6:语料归一化 + chat template 前置验死
  merge_lora.py            工单 §7.1/§7.2:adapter 合并回 bf16 checkpoint
  mp_select.py             工单 §4.1:按路由计数挑热专家 + 位宽账(--bits-table)
verify/
  test_core_cpu.py         内核自测,24 项,纯 CPU
  test_fakequant_cpu.py    端到端自测,合成 checkpoint
  test_offline_ptq_cpu.py  PTQ 自测,55 项,含 2bit 三档对比
  test_merge_lora_cpu.py   合并自测,22 项,纯 CPU,参考值用 einsum 独立算
  test_mp_select_cpu.py    混合精度自测,60 项,含位宽口径与 --keep-hot 端到端
  test_adapter_save_cpu.py adapter 落盘自测,15 项,只需 torch + safetensors
  probe_lora_tree.py       工单 §0.1:LoRA 能否挂上 Qwen 的配置树(需训练环境,流程见"§0.1 闸门怎么跑")
  probe_recipe_cpu.py      训练配置能否 import / 构造(需训练环境;安装脚本的闸门 5)
  probe_parallelize_cpu.py 8 个 CPU 进程把整个模型真的并行化一遍(安装脚本的闸门 6)
  probe_triton_launcher_cpu.py  Triton kernel 的 C++ 启动代码在本机编不编得出来(闸门 7)
  probe_weights_map_cpu.py checkpoint 的键和形状与模型是否对得上(要 checkpoint 目录,起训练前跑)
```

`src/unisvq_core.py` 是内部 UniSVQ 参考实现的 vendoring 副本,相对原版只改了
位宽接线(`UNISVQ_NBITS`,2–8 可配;原版把 2 写死成模块常量)并补了几个辅助函数,
改动在文件头有记录。

## 进度

工单的编号约定:**`§N.M` 指"阶段 N 的第 M 行工单"**;叙述章节用字母
(§B 前提、§C 设计决定、§F 风险登记、§G 开工前确认 等)。2026-11-06 改过 ——
在那之前顶层章节也用 0..7,`§2.2` 和 `§6.1` 各有两个互不相干的含义。

| 工单项 | 状态 |
|---|---|
| 开发工单 | **完成** |
| §B.5 量化范围确认 | **完成**,只做路由专家 |
| §2.1 `N_BITS` 接环境变量 | **完成** |
| 内核 vendoring + 自测 | **完成**,24 项全过 |
| §1.1 `fakequant_rtn.py` | **完成**,端到端自测全过 |
| §0.1 LoRA 配置树检查 | **通过**(2026-11-16,开发机上一个全新的 3.12 环境 + 干净 clone + 兼容层,9 条判据全 OK)。另一台机器上待复跑 |
| §0.2 下权重 / §0.3 形状预检 | 未做 |
| §2.2 `offline_ptq.py` | **代码完成**,55 项自测全过;真激活未跑过 |
| §2.3 单层 dry-run | 合成数据上过了;**真权重 + 真激活未做** |
| §2.4 全量 40 层量化 | 未做,等真权重 + 真激活 |
| §2.5 `write_dequant_ckpt.py` | **不写了** —— `offline_ptq.py run` 直接写出完整 bf16 HF 目录,没有中间产物;只差一个抽查脚本 |
| §4.1 `mp_select.py` 挑热专家 | **代码完成**,60 项自测全过;真激活未跑过 |
| §4.2 `--keep-hot` 跳过热专家 | **代码完成**,保留专家逐位直通已验 |
| §4.3 混精评测 | 未做,卡被占 |
| §6.1–6.5 LoRA trainer 配置 | **代码完成**(`qwen36_lora_recipe.py`);三个配置在 CPU 上能构造(闸门 5);**2026-11-17 在另一台机器 8 卡上跑通 5 步 smoke**(原始权重 + 合成语料) |
| §6.5 adapter 落盘 | **改为本仓库自带**(`qwen36_adapter_save.py`),21 项自测全过;smoke 里 8 个 rank 各落一片、能拼成 200 个全局张量。第一批分片的键带 `_checkpoint_wrapped_module`,已改成落规范名字 |
| §6.6 语料 + chat template | **完成**,两条路径都冒烟过 |
| §7.1/§7.2 `merge_lora.py` | **完成**,26 项自测全过;读分片时会规范化参数名(旧分片也能用) |
| §6 LoRA 训练本身 | 5 步 smoke 已跑通;**正式训练未做** —— 还差量化后的基座、真实语料、基线评测 |

### LoRA 线的三件代码就位之后,还差什么

1. **训练环境**(`env/2_pip_train.sh`)—— 2026-11-16 在开发机的一个全新环境里
   **七道闸门全过**(CANN 8.5.1、GCC 10.3.1,纯 CPU)。另一台机器上闸门 1–6 已过;
   第 7 道是 2026-11-17 第一次前向之后加的,待复跑。
2. **§0.1 闸门**:`verify/probe_lora_tree.py` —— 同上,已过。
3. **一张空闲卡**跑 `qwen36_35b_a3b_lora_smoke`(20 步)。**这一步还没人跑过。**

闸门 4/5 验到的是:converter 能把 40 层路由专家换成 LoRA 版、一层专家模块能
实例化且 adapter 参数量对得上解析值、三个训练配置能构造。**没验到的**是真权重
加载、前向、反传、分布式切分和 adapter 实际落盘 —— 这些要第 3 步。

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

### 十个会改排期的发现

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

**六、DSv4.1 那个 2.609 bit 是 int8 口径,本路线照抄保留比例会变成 3.42 bit。**

`model_cfg.py` 的 `bits(K) = N_BITS + (8 - N_BITS)·K/E` —— 保留的专家存 **int8**。
本路线全程 bf16(工单 §C.1),保留的是 16 bit:

| 配置 | 保留比 | 平均位宽 | 产物 |
|---|---:|---:|---:|
| DSv4.1 hot39/384 保 int8(**65.66 GPQA 就是这个点**) | 10.16% | 2.609 bit | — |
| 同保留比例 hot26 保 bf16 | 10.16% | **3.422 bit** | 13.78 GB |
| 同位宽预算 hot11 保 bf16 | 4.30% | 2.602 bit | 10.48 GB |

所以"2bit + hot26 回退"在本路线是个 **3.42 bit** 的模型。拿它的分数和 65.66 比,
是拿高了 0.81 bit 的模型去比。要同预算比,用 **hot11**。
`python src/mp_select.py --bits-table --experts 256 --bits 2` 直接打这张表。

顺带两件:bf16 保留让 2bit 产物从 8.05 GB 涨到 13.78 GB(几乎翻倍);
3bit 基座下 DSv4.1 **没有**混精的公开数,所以那一行只能当推算值,不能当靶子。

**七、工单是照同事的工作区核的,不是照已提交的代码 —— 干净 clone 上差两处。**

2026-11-16 第一次在一份干净的 `torchtitan-npu@3f9f99a` 上跑闸门 0.1,报
`_LoRAModelConfig.__init__() missing 4 required keyword-only arguments: 'hc_head', ...`。
逐条核对本仓库的 import 之后,确认有两样东西**只存在于 `l84414662` 的工作区、从未提交**:

| 缺什么 | 在同事工作区里的状态 | 干净 clone 上的后果 | 本仓库的处理 |
|---|---|---|---|
| `lora.py` 里按实际根配置类派生 `_LoRAModelConfig` 的修复(27 行) | 已修改、未提交 | `convert()` 最后一行把根配置派生成写死的 DSv4 类,Qwen 直接 TypeError | `src/qwen36_lora_compat.py`:只在 `convert()` 期间替换那个模块全局,不改 torchtitan-npu |
| `patches/torchtitan/models/common/adapter_save.py` | 未跟踪文件 | 训练能启动、加载完 70 GB 权重、建好优化器,然后才 `ModuleNotFoundError` | `src/qwen36_adapter_save.py`:自带一份,落盘格式与 `merge_lora.py` 的读法对拍 |

TypeError 出在 `convert()` 的**最后一行**:前面的树遍历、后缀匹配、strict 校验都已通过。
所以"分组专家 LoRA 能复用到 Qwen"这个判断本身是对的,错的是"一个文件都不用改"
这句话的前提。同事工作区里对 upstream torchtitan 还有两处未提交改动
(`text_datasets.py` 加了三个 DSv4 语料、`protocols/module.py` 为 int8 基座保留
`requires_grad`),本路线都用不到。

同一次实跑还查出:训练入口 `torchtitan_npu.models.qwen3_5` 需要 `torchvision`
(闸门 0.1 的入口是 upstream 的 `torchtitan.models.qwen3_5`,走不到),
以及 probe 自己的三处错 —— 按 `__dict__` 遍历 slots 配置所以一个节点都数不到、
按类名里有没有 "LoRA" 判断所以全是假阴性、把结论性失败报成环境问题。都已修。

**八、照原来的写法,基座权重根本不会被加载 —— 而且不报错。**

2026-11-16 准备跑 §0.4、核对"`HF_ASSETS_PATH` 到底该填什么"时发现。三处都用
`git show` 在已提交的代码上核过(upstream `086bf6c16`、torchtitan-npu `3f9f99a`):

| 问题 | 依据 | 后果 | 本仓库的处理 |
|---|---|---|---|
| `checkpoint.enable` 默认 `False`,上游 `qwen35_35b_a3b()` 没开,本仓库的 recipe 也只改了 `interval` | `components/checkpointer/base.py`:`load()` 第一行是 `if not self.enable: return False` | 权重不加载,基座随机初始化;训练正常启动、loss 照降、`exit=0`,**一条告警都没有** | `_wire_weights`:`enable=True`、`initial_load_in_hf=True`;同时 `load_only=True`,否则最后一步会强制存一份约 70 GB 的全模型 |
| `HF_ASSETS_PATH` / `INITIAL_LOAD_PATH` / `CKPT_SAVE_LOAD_PATH` 这些环境变量**没有任何人读** | `scripts/run_train.sh` 只认 `NGPU` / `MODULE` / `CONFIG`;只有 `examples/` 下的脚本把 `HF_ASSETS_PATH` 转成 `--hf-assets-path` | tokenizer 和权重去找上游默认的 `./assets/hf/Qwen3.5-35B-A3B` | recipe 自己读 `HF_ASSETS_PATH`(必填、必须绝对路径,否则拒绝构造);`INITIAL_LOAD_PATH` 可选 |
| NPU 的 GDN kernel override 是 `examples/qwen3_5/*.sh` 用 `--override.imports` 加的,配置工厂不带 | `examples/qwen3_5/readme.md`:"配置工厂…不会静默注入 NPU 实现";上游 `GatedDeltaKernel` 默认后端是 `fla_chunked` | 直接调 `scripts/run_train.sh` 时走的是 FLA 自己的 kernel。**它在 NPU 上能不能跑没有验证过**,只知道 examples 不走它 | recipe 把 `torchtitan_npu.override.qwen3_5.gated_delta.npu` 加进 `override.imports`,跟齐 examples |

第一条和 DSv4.1 不设 `UNISVQ_CKPT` 是同一种失败:跑完几个小时,得到一份补偿随机基座的
adapter。它之所以一直没露头,是因为到这一步之前的所有闸门都不加载权重。

连带更正:安装脚本结尾原先打印的 dry run 命令**是错的** —— `CONFIG` 写成了 27B 的
`qwen35_27b_4k_sft`(不是这个模型)、没给 `DATA_FILES`(构造配置时就 `KeyError`)、
`NGPU=1`(35B 的 EP=8,单卡也装不下)、路径用的是没人读的环境变量。本节上面的
冒烟命令原先也写着 `NGPU=1`。都已改成"LoRA 训练怎么起"里那一条。

**九、第一次上卡:upstream 对"Qwen + LoRA"和"Qwen3.6 的真文件"各有一处不成立。**

2026-11-16 另一台机器上 8 卡 dry run,`Trainer.__init__` 走到并行化报:

```
ValueError: LoRA_ClampGroupedExperts.w13_lora_a has no placement declared
in sharding_config.state_shardings.
```

顺着这一步往后在 CPU 上核(开发机的 16 个 NPU 当时全被占着,没法上卡),查出三处:

| 问题 | 依据 | 后果 | 本仓库的处理 |
|---|---|---|---|
| Qwen 的专家切分声明表里没有 adapter | 并行化要每个参数都有声明。DSv4 的表(`deepseek_v4/sharding.py`)把 5 个 adapter 和 3 个基座权重列在一起;upstream `qwen3_5/sharding.py` 的表只有 `w1_EFD` / `w2_EDF` / `w3_EFD` | 上面那个 `ValueError`,在加载权重之前 | `qwen36_lora_compat.ensure_adapter_sharding`:把 DSv4 表里的 adapter 条目原样抄进 Qwen 的表 |
| **路由专家权重的排布,upstream 假设的是真文件的转置** | upstream 适配器按 `gate_up_proj [E, dim, 2·hidden]` / `down_proj [E, hidden, dim]` 读(和 transformers 里更早一代的 `Qwen3VLMoeTextExperts` 一样)。Qwen3.6-35B-A3B 真文件(读 26 个分片的文件头)是 `(256, 1024, 2048)` / `(256, 2048, 512)`,即 `[out, in]` | 40 层 × 2 = 80 个张量形状对不上,权重加载不了。这正是工单 §0.4 要拦的"上游 flavor 加载不了真权重" | `src/qwen36_state_dict_adapter.py`:在 upstream 适配器外面套一层,只对这两类键各做一次转置 |
| MoE 负载均衡偏置默认开着 | upstream `MoE.Config.load_balance_coeff = 1e-3`:每层一个 `expert_bias_E`(初值 0),优化器 pre-hook 每步调整,并加到路由打分上 | 训练时的路由一步步偏离部署时的路由(256 步最多漂 0.256);这个偏置不在 checkpoint 里,合并时也不会导出 | recipe 里设成 `None`(`_disable_load_balancing`)。**这是训练语义上的选择,不是 bug 修复**,不同意可以去掉那一行 |

**`gate_up_proj` 里哪一半是 gate —— 定了:前一半。** 依据是 vLLM 的
`model_executor/models/qwen3_5.py::load_weights`(基线评测就是它加载的这份 checkpoint):

```python
# qwen3.5 no need to transpose
# loaded_weight = loaded_weight.transpose(-1, -2)
if "experts.gate_up_proj" in name:
    loaded_weight = loaded_weight.chunk(2, dim=-2)     # [0] -> w1, [1] -> w3
```

所以 `--gate-up-order` 的默认值 `gate_first` 是对的,那个未决问题可以关了。
本仓库自己的 `fakequant_rtn.py` / `offline_ptq.py` / `merge_lora.py` 一直是按真文件的
排布写的(并且自动判朝向),不受这一条影响;受影响的只有"训练框架加载权重"这一环。

新增两个探针,都不占 NPU:

* `verify/probe_parallelize_cpu.py`(安装脚本的闸门 6):起 8 个 CPU 进程,照
  `Trainer.__init__` 的顺序把 35B 的完整配置建在 meta device 上并行化。不带修复时
  它报的就是上面那一句;带修复时 8 个进程各持有 32/256 个专家,可训张量恰好 200 个。
* `verify/probe_weights_map_cpu.py`:用真 checkpoint 的索引和分片文件头核对整张权重
  映射。实测(Qwen3.6-35B-A3B):模型要的 1026 个键全在、形状全一致;1294 个非 adapter
  状态全部会被加载;checkpoint 里没用到的只有 19 个 `mtp.*`(多 token 预测头,
  模型没有这个分支)。把适配器换回 upstream 的再跑,报 80 个张量形状不一致。

**这一条为什么之前没拦住**:闸门 4 / 5 只到"单进程、不建设备网格、不碰 checkpoint"为止。
工单 §0.3 的形状预检核的是 config 推出来的形状,没有走 upstream 的适配器。

**十、第一次前向:GDN kernel 的启动代码在这套环境上编不过。**

2026-11-17,权重加载成功(60.85 秒,三行日志都对)、跑到第一次前向,第 0 层的
GatedDeltaNet 报:

```
RuntimeError: Failed to compile .../precompiled.h.gch
  ATen/ATen.h:5:2: error: #error C++20 or later compatible compiler is required to use ATen.
  AclInterface.h:308:5: error: 'aclmdlRICondHandle' has not been declared
```

Triton-Ascend 每个 kernel **第一次运行时**才现编一小段 C++ 启动代码。这套环境上三处对不上:

| 问题 | 依据 | 本仓库的处理 |
|---|---|---|
| triton-ascend 3.2.1 写死 `-std=c++17`,torch 2.14 的 `ATen.h` 要求 C++20 | `triton/backends/ascend/utils.py`;torchtitan-npu 自己的测试也把这一条标成已知问题(`xfail`:"Re-enable once Triton-Ascend is updated") | 换成 `-std=c++20` |
| GCC 10 开了 C++20,`__cplusplus` 仍是 `201709L`(GCC 11 起才是 `202002L`),`ATen.h` 的检查还是不过 | `echo \| g++ -std=c++20 -x c++ -E -dM -` | 显式定义成 `202002L`。头文件本身在 GCC 10 的 C++20 模式下编得过(实测) |
| torch_npu 的头文件引用了 CANN 9.x 才有的类型,CANN 8.x 的头里没有 | `AclInterface.h` 声明的三个函数用到 `aclmdlRICondHandle` / `aclmdlRICondTaskParams` | `env/triton_cann_compat.h` 补两个声明。启动代码只 include 那个头,不调用这三个函数 |

做法是一个**编译器包装脚本** `env/triton_cxx.sh`:triton-ascend 取编译器时读环境变量 `CC`,
`src/qwen36_triton_env.py`(recipe 导入时调)把 `CC` 指过去。不改任何第三方包;只对带
`-std=c++17` 的那几条命令动手,其余原样交给真正的 `g++`。`CC` 已被设成别的东西时不覆盖,
只告警(要两者都用:`QWEN36_REAL_CXX=<你的编译器> CC=<本仓库>/env/triton_cxx.sh`)。

`verify/probe_triton_launcher_cpu.py`(安装脚本的闸门 7)把 triton-ascend 自己的那两步原样
调一遍:预编译头、编启动代码的 `.so`,再 import 它。不占 NPU。开发机(GCC 10.3.1、
CANN 8.5.1)实测:不带包装时报的就是上面那几行;带包装三步全过。预编译头约 500 MB,
第一次会写进 `~/.triton/cache/`。

**根子上的问题没有解决,只是绕过去了:这台机器的 CANN 比这套软件栈验证过的版本老一个
大版本。** torchtitan-npu 的 `docs/user-guides/installation.md` 写的是 torch 2.14 配
CANN 9.2.0,而开发机和另一台机器都是 8.x(安装脚本一开始就在告警)。torch_npu 是按新
CANN 编的,在老 CANN 上 import 得起来是因为符号是用到才解析的 —— 所以同类问题只会在
"第一次用到某个东西"时冒出来。**已知还没走到的一处**:10 层全注意力层(第一层在
layer 3);另一台机器上安装脚本的 CANN 检查里 `aclnnFlashAttentionScoreV3` 的符号数是 0,
它会不会被用到没有验证过。根治办法是换 CANN 9.x。

**验到哪一步**:闸门 5 现在核对这些开关在配置里是对的,并且把训练启动器的参数解析
原样走了一遍(`--training.steps 5` 生效、开关没被冲掉)。**真权重是否加载成功、
加载后键是否全对上,只有上卡的 dry run 日志能回答。**

## 下一步

与卡的可用性解耦,按这个顺序:

1. ~~下权重、核对键名分类、3bit 假量化~~ —— **已完成**,7.1 分钟,
   SNR min 10.70 / p50 11.12 / max 11.68 dB。
2. **不需要卡**:建训练环境(`env/2_pip_train.sh`,proxy 要关),
   跑 `verify/probe_lora_tree.py`(§0.1)。**这是整条 LoRA 线的闸门。**
   完整流程和 torchtitan 的来历见"§0.1 闸门怎么跑" —— upstream torchtitan
   要单独 clone 进 `third_party/`,它不是 submodule。
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
8. 2bit 混精:`mp_select.py` 挑热专家 → `offline_ptq.py --keep-hot`。
   **这一步必须有真激活**(路由计数没有 H=I 那种免校准退路),但它要的
   token 数比 Hessian 少得多 —— 计数稳定比 `xe^T xe` 稳定容易。
   先定用 hot11(同位宽预算)还是 hot26(同保留比例),见发现六。

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

## 混合精度怎么挑

```bash
python src/mp_select.py --bits-table --experts 256 --bits 2
```

先看这张位宽账再动手 —— 选 hot11 还是 hot26 决定了产物能不能和 DSv4.1 的数比
(发现六)。然后:

```bash
python src/mp_select.py --in <ckpt> --acts acts/ --out keep_hot11.json -k 11
python src/offline_ptq.py run --in <ckpt> --out <dst> --bits 2 --acts acts/ --keep-hot keep_hot11.json
```

`mp_select.py` 同时出 `hot` 和 `cold` 两个变体(cold = 校准 token 最少的 K 个,
Hessian 样本最少、2bit 误差最大;是个便宜的对照假设,不是主路径),
外加一个 `*_info.json` 记两种口径的平均位宽和并列打破方式。

`--keep-hot` 会挡掉三类静默失败:专家号越界、某层保留了全部专家、
值不是整数列表(把 `*_info.json` 当 keep 传)。**要量化的层不在 keep 里**
只警告不拦,但会明说那层整层走低位宽 —— 这条原版列为"缺层"自检。

## LoRA 训练怎么起

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python src/prepare_corpus.py --tokenizer <checkpoint目录> --synthetic 64 --out <工作目录>/smoke.jsonl
```

```bash
cd ~/src/torchtitan-npu && NGPU=8 MODULE=qwen36_lora_recipe CONFIG=qwen36_35b_a3b_lora_smoke HF_ASSETS_PATH=<checkpoint目录> DATA_FILES=<工作目录>/smoke.jsonl ADAPTER_SAVE_DIR=<工作目录>/adapter ADAPTER_SAVE_EVERY=20 bash scripts/run_train.sh
```

所有路径都要**绝对路径**(训练是 cd 到 torchtitan-npu 里起的)。

**`<checkpoint目录>` 是 HF 格式的模型目录,一个目录管三件事**:tokenizer
(`tokenizer.json`、`tokenizer_config.json`、chat template)、权重索引
(`model.safetensors.index.json`)、权重分片。没有单独的"tokenizer 目录"。

| 跑什么 | `HF_ASSETS_PATH` 填 |
|---|---|
| §0.4 的 5 步 dry run(只验"真权重能加载") | 原始下载的 Qwen3.6-35B-A3B 目录;命令末尾加 `--training.steps 5` |
| 冒烟 / 交付训练 | `fakequant_rtn.py` 或 `offline_ptq.py` 的**输出目录**(它们会把 tokenizer、`config.json`、索引原样带过去)—— adapter 要补偿的是那一份量化误差 |

**`NGPU` 必须是 8**:配置里 `expert_parallel_degree=8`,而且 bf16 全模型约 70 GB,
一张 60 GB 的卡装不下。

这几个环境变量(`HF_ASSETS_PATH` / `DATA_FILES` / `ADAPTER_SAVE_DIR` /
`ADAPTER_SAVE_EVERY`,以及可选的 `INITIAL_LOAD_PATH`)是 **`qwen36_lora_recipe`
自己读的**。`scripts/run_train.sh` 只认 `NGPU` / `MODULE` / `CONFIG`,换成别的
MODULE 时 `HF_ASSETS_PATH=...` 这种写法不会生效,也不会报错。

**启动后先看日志里这三行**(都出自上游 `components/checkpointer/dcp.py`):

| 日志 | 含义 |
|---|---|
| `Loading HF safetensors from --model.hf_assets_path: <目录>` | 开始加载真权重 |
| `Finished loading the checkpoint in N seconds.` | 加载完成 |
| `No checkpoint was provided, this is a fresh start.` | **没加载,基座是随机的** —— 立刻停 |

**语料的形状(2026-11-16 用真的 Qwen3.6 模板核对)。** 上游 `ChatDataset` 只收单轮
`[user, assistant]`,多轮或带 system 的样本会在第一个 batch 抛 `ValueError`。
`prepare_corpus.py` 现在用和上游**逐项相同**的 jinja 环境渲染,并提前拦掉不合形状的
样本(原先用的是严格模式,会在模板的 `{%- if tools ... %}` 上误报
`'tools' is undefined`,而训练本身并不会)。

模板若带 `{%- generation %}` 标签(Hub 上较新的版本有),上游的 jinja 环境不认,
训练建 tokenizer 时会报 `TemplateSyntaxError`;`prepare_corpus.py` 会先报出来并说明处理办法。

**未决**:用 Qwen3.6 的模板渲出来,prompt 段到 `<|im_start|>assistant\n<think>\n` 为止,
算 loss 的部分是 `\n</think>\n\n<回答><|im_end|>` —— 即**每条样本都在教模型输出一个空的
思考块**。冒烟无所谓;交付训练前要确认这和评测时的思考开关是否一致。

`MODULE=qwen36_lora_recipe` 不是 `torchtitan_npu.models.qwen3_5` —— 那个模块的
全部名字都被搬进了我们这个模块,所以它是**超集**,而 torchtitan-npu 的
clone 一个文件都不用改。在一份**干净的** clone 上这句话要靠本仓库的
`qwen36_lora_compat.py` 和 `qwen36_adapter_save.py` 才成立 —— 见"十个会改排期的
发现"第七至十条。起训练之前先确认闸门 5、6、7 过了,并核对权重映射:

```bash
python3 verify/probe_recipe_cpu.py
```

```bash
python3 verify/probe_parallelize_cpu.py
```

```bash
python3 verify/probe_triton_launcher_cpu.py
```

```bash
python3 verify/probe_weights_map_cpu.py --hf-assets <checkpoint目录>
```

第三条要真的 checkpoint 目录:它读索引和各分片的**文件头**(不读权重数据,十几秒),
核对模型要的每个键 checkpoint 里都有、形状一致、没有哪个模型参数被漏掉。
**换了 checkpoint(比如假量化的输出)就重跑一次。**

训完合并:

```bash
python src/merge_lora.py --base <假量化ckpt> --adapter <adapter输出目录> --rank 16 --alpha 32 --dry-run
```

**每次训练(包括 smoke)之后都先跑一遍 `--dry-run`。** 它是"训练保存的格式"和"合并读的格式"
唯一一次在真产物上对接。2026-11-17 第一批真分片就在这里报了:

```
!! 有 200 个 lora_ 张量不在路由专家上 —— 训练配置和本脚本不匹配。
   例:layers.0._checkpoint_wrapped_module.moe.routed_experts.inner_experts.w13_lora_a
```

训练时 activation checkpoint 给每一层套了一层包装,`named_parameters()` 给出的参数名中间
多一段 `_checkpoint_wrapped_module`,而规范名字(state_dict / HF checkpoint 那边)没有。
保存器原样写了出去,合并工具只认规范名字。adapter 本身没问题。两头都改了:
保存器现在落规范名字;合并工具读的时候也规范化一遍,所以那批旧分片不用重训。

同一处还加了一个守卫:专家参数的切分是 `(_StridedShard(dim=0, sf=<EP>), Shard(dim=0))`,
只有 **rank 数 == EP**(我们的 8 卡 / EP=8)时才能按 rank 顺序直接拼;否则各 rank 持有的
专家是交错的,直接拼形状照样对得上、顺序是错的 —— 现在会拒绝。
`verify/probe_parallelize_cpu.py`(闸门 6)用并行化之后的真 DTensor 把名字和切分方式
同合并工具对一遍,不用等训完才发现。

`--alpha` **故意没有默认值**:adapter 分片(`qwen36_adapter_save.py`,格式同
DSv4.1 的 `adapter_save.py`)的 metadata 里那个
`rank` 是**进程 rank**,alpha 根本没存进去。DSv4.1 就是在这里把 delta
放大了 8 倍(训练日志 alpha=4,评测照默认值传了 32),那批 `3bit+LoRA`
参照数因此全部不可用。合并后的 `merge_info.json` 和分片 metadata 里会
记下实际用的 scaling,评测端照它核对。

## §0.1 闸门怎么跑(从零开始的完整流程)

`verify/probe_lora_tree.py` 验的是:DSv4 的分组专家 LoRA 工厂能不能挂上
Qwen3.5/3.6-35B-A3B 的配置树。**不需要权重、不需要卡**,但需要 torchtitan
能 import —— 而 torchtitan 不是一个 `pip install` 就有的东西,见下一节。

### 第 0 步:先搞清 torchtitan 从哪来

这个项目依赖**两个**仓库,它们是嵌套关系但**要分别拿、分别装**:

| | 包名 | 来源 | 核对过的版本 | Python |
|---|---|---|---|---|
| Ascend 插件 | `torchtitan_npu` | `https://gitcode.com/cann/torchtitan-npu.git` | `3f9f99a`(0.1.0) | **>= 3.12** |
| upstream | `torchtitan` | `https://gitcode.com/GitHub_Trending/to/torchtitan.git` | `086bf6c16`(0.3.0) | >= 3.11 |

**"核对过的版本"这一列必须照做,两个仓库都要 checkout 到表里的 commit。**
torchtitan-npu 的 master 一直在走:2026-11-14 的 `cbad953` 比 `3f9f99a` 多 166 个提交,
`requirements.txt` 已经换了一整套 —— 正式版 `torch==2.14.0` / `torch_npu==2.14.0`
(不是这里用的 `dev20260719` / `dev20260808`)、`attn-gym==0.0.16`(不是 `0.0.9`),
CANN 也升了。master 配本项目装的那套包会在 `import torchtitan_npu` 时报
`No module named 'attn_gym.sparse.gather_attn'`。本项目的 LoRA 配置树和 state_dict
键映射都是照 `3f9f99a` 核的。安装脚本会自动切过去(工作区有未提交改动时停下不动);
确实要跟 master 就 `TTNPU_PIN=none`,并自己照它的 `requirements.txt` 对齐全部版本。

upstream 要放在插件仓库的 `third_party/torchtitan/` 下。**三个事实合在一起,
就是"常规方法装不上"的全部原因:**

1. `third_party/` 被 torchtitan-npu 自己的 `.gitignore` 挡掉(第 115 行 `/third_party/`),
   所以 **clone 下来这个目录是空的**。
2. 它**不是 submodule** —— `.gitmodules` 里只有 `third_party/agent-skills/KPBot` 一条。
   `git submodule update --init` 退出码 0,但拿不到 torchtitan。
3. `pip install -e <torchtitan-npu>` **只装 `torchtitan_npu`**:它的 `pyproject.toml`
   里 `packages.find` 写着 `exclude = ["third_party*"]`。`dependencies` 里虽然列了
   `torchtitan`,但 PyPI 上那个不是这里要的版本,而且本流程一律 `--no-deps`。

所以只装了插件的环境里,`import torchtitan_npu` 能过,`import torchtitan` 报
`No module named 'torchtitan'` —— 看着像漏装一个包,实际是少 clone 一个仓库。

另外 torch 是钉死的 nightly:`torch==2.14.0.dev20260719`、`torch_npu==2.14.0.dev20260808`
(`torchtitan-npu/requirements.txt`),见下面"训练环境安装"的版本表。

### 第 1 步:建环境(Python 3.12)

```bash
bash env/1_conda_envs.sh
```

```bash
conda activate lqq-train
```

**是 `lqq-train`,不是 `lqq-quant`。** 两个名字像,但 `lqq-quant` 是量化线的环境
(Python 3.10),训练线的包装不进去。激活错了 `env/2_pip_train.sh` 会在第一步
停下并指路,不会再跑到 GATE 1 才报一句看不出原因的
`is not a supported wheel on this platform`。

Python 必须 >= 3.12。在 3.9 上 import 会报
`dataclass() got an unexpected keyword argument 'kw_only'` —— 那是 Python 版本
(`kw_only` 是 3.10 才有的),不是 torchtitan 坏了。

### 第 2 步:装包。两条路选一条

**A. 用脚本(推荐)。** 它会 clone 两个仓库、钉 commit、`--no-deps` 装两个包,
然后依次过四道闸门,第 4 道就是本脚本:

```bash
bash env/2_pip_train.sh
```

torchtitan-npu 默认放 `~/src/torchtitan-npu`,要换位置就 `TT_DIR=<目录> bash env/2_pip_train.sh`。

**B. 手动。** 机器上已经有一份 torchtitan-npu、或者不想跑整个脚本时:

```bash
git clone https://gitcode.com/cann/torchtitan-npu.git ~/src/torchtitan-npu
```

```bash
git -C ~/src/torchtitan-npu checkout 3f9f99a
```

```bash
git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git ~/src/torchtitan-npu/third_party/torchtitan
```

```bash
git -C ~/src/torchtitan-npu/third_party/torchtitan checkout 086bf6c16
```

```bash
pip install --no-deps -e ~/src/torchtitan-npu/third_party/torchtitan
```

```bash
pip install --no-deps -e ~/src/torchtitan-npu
```

**`--no-deps` 不能省。** torchtitan 钉着 torch nightly,让 pip 自己去解依赖,
它会把你装好的 torch / torch_npu 换掉,而这两个必须成对(见下面的版本表)。
torch、torch_npu 和其余依赖的装法照 `env/2_pip_train.sh` 第 1–3 段。

### 第 3 步:跑

```bash
python3 verify/probe_lora_tree.py
```

两个包都装进当前环境了就这样直接跑。**没装、只想指一个现成的目录**也行 ——
脚本会自己把 `<TT_ROOT>` 和 `<TT_ROOT>/third_party/torchtitan` 接上 `sys.path`,
不用手配 `PYTHONPATH`:

```bash
TT_ROOT=<torchtitan-npu 目录> python3 verify/probe_lora_tree.py
```

可调的环境变量:`TT_FLAVOR`(默认 `35B-A3B`)、`TT_RANK`(16)、`TT_ALPHA`(32)。
**不要用 `python -I` 跑它** —— `-I` 会忽略 `PYTHONPATH` 并去掉 user site,
包如果是 `pip install --user` 装的就直接找不到。

### 第 4 步:读结果

开头三行先告诉你它用的是哪套东西,对不上就不用往下看了:

```
torchtitan root: /home/.../torchtitan-npu
Python: 3.12.x
torchtitan:     /home/.../third_party/torchtitan/torchtitan/__init__.py
converter:      torchtitan-npu 的 lora.py 是已提交的原版(根配置写死为 DSv4) -> 用本仓库 qwen36_lora_compat 的兼容层
```

`converter:` 那行在干净 clone 上就该是上面这样。没给 `TT_ROOT` 且两个包都已装进
环境时,probe 用装好的那份,不再去猜目录。

通过的样子是"判据"段九条全 `OK`(下面是 2026-11-16 的实跑输出):

| 判据 | 含义 | 实测 |
|---|---|---|
| 分组专家被 LoRA 覆盖 | 转换后 `inner_experts` 的配置类继承了 `LoRAOptions` | `grouped=40 / 40` |
| 命中数 == 层数 | 40 层每层都挂上了 | `40 == 40` |
| dense 命中为 0 | `target_modules=()` 生效,LoRA 只加在路由专家上 | `dense=0` |
| converter 自己报的计数一致 | 截它那行日志对拍 | `dense=0 grouped=40 rank=16 expert_rank=16` |
| 根配置带 lora 字段,且仍构造原模型类 | 兼容层派生的根配置没换掉模型 | `Qwen35Model` |
| 实例化一层:adapter 参数名和形状 | `w13_lora_a` / `w1_lora_b` / `w3_lora_b` / `w2_lora_a` / `w2_lora_b` | 5 个张量 |
| 实例化一层:adapter 参数量 | 等于 `E × r × (2·dim + 3·moe_inter)` | `23068672`(每层 23.07 M,全模型 0.923 G) |
| 实例化一层:基座权重仍在 | `w1_EFD` / `w2_EDF` / `w3_EFD` | `[256,512,2048]` / `[256,2048,512]` / `[256,512,2048]` |
| scaling == alpha / rank | | `2.0` |

"是不是 LoRA 配置"按是否继承 `LoRAOptions` 判,**不按类名**:工厂生成的配置类叫
`_make_lora_adapter_config_cls.<locals>.Config`,名字里没有大写的 LoRA。

失败分两种,**含义完全不同**:

| 现象 | 性质 | 怎么办 |
|---|---|---|
| 退出码 2,`!! 失败于 [import ...]` | **环境问题,不是结论** | 脚本会打出诊断:Python 版本 / third_party 是空的 / 路径没找到 / 缺哪个包,并给出修法。修完重跑 |
| 退出码 1,`!! 失败于 [model_registry(...)]` | **代码层面的结论** | import 已全过,是 converter 在 Qwen 的配置树上没走完。重装环境没用 |
| 退出码 1,判据段出现 `X` | **结论为假** | 转换走完了但结果不对。阶段 6/7 的排期作废,要改 converter |

**不要把第一种记成第二种。** import 期炸掉什么都没验到。

常见的 import 失败对照:

| 报错 | 真因 |
|---|---|
| `No module named 'torchtitan'` | upstream 没 clone 进 `third_party/`,或没装(第 0 步的三个事实) |
| `No module named 'torchtitan_npu'` | 插件没装,且没给 `TT_ROOT` |
| `dataclass() got an unexpected keyword argument 'kw_only'` | Python < 3.10。要 3.12 |
| `cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'` | torch 太旧,要 2.14 nightly |
| `Mismatch in kernel C++ signatures` | torch 和 torch_npu 日期不成对 |

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
| 前置 | Python 是 3.12、架构是 aarch64 | 在 `lqq-quant`(3.10)里跑了本脚本。不过就直接停,什么都不装 |
| 前置 | pip 索引取得到包(真下载一次 `filelock`,不装) | 镜像/代理状态不对。不探的话会滑到闸门 1 才露头,见下面"索引不通长什么样" |
| 1 | torch 装上且能 import | 直链被回收 |
| 2 | **torch_npu 能 import** | ABI 不符 / CANN 找不到 |
| 3 | torchtitan_npu 能 import | torch 太旧(缺 `DataParallelMeshDims`);或 upstream torchtitan 没 clone 进 `third_party/`(它不是 submodule);或 torchtitan-npu 没钉在 `3f9f99a`(报 `No module named 'attn_gym.…'`) |
| 4 | 分组专家 LoRA 挂上 Qwen3.5-35B-A3B | 这是 LoRA 线的闸门。单独跑和读结果见上面"§0.1 闸门怎么跑"。缺 `fla` / `triton` 会在它之前被拦下 |
| 5 | 本仓库的训练配置能 import、能构造,**且配置里"加载真权重"是开着的**(`verify/probe_recipe_cpu.py`) | 缺 `torchvision`:训练入口 `torchtitan_npu.models.qwen3_5` 要它,闸门 4 的入口走不到。或 recipe 被改回了不加载权重的写法(见"八个会改排期的发现"第八条) |
| 6 | 8 个 CPU 进程把整个模型(meta device)并行化一遍(`verify/probe_parallelize_cpu.py`) | 某个模块的参数没有切分声明(第一次上卡报的 `w13_lora_a has no placement declared` 就是这一类)。前面几道都是单进程,走不到这里 |
| 7 | Triton kernel 的 C++ 启动代码在本机编得出来(`verify/probe_triton_launcher_cpu.py`) | 编译器 / CANN 头文件和 torch、torch_npu 的头文件对不上。不过的话 GDN kernel 跑不起来,而且要等到第一次前向才报。见"十个会改排期的发现"第十条 |

全程不占 NPU(闸门 4 置空 `ASCEND_RT_VISIBLE_DEVICES`,纯 CPU 跑配置树)。
真正上卡的 5 步 dry run 是脚本最后单独给出的几条命令(要 8 张卡),见"LoRA 训练怎么起"。

**索引不通长什么样。** 旧版脚本上的表现是闸门 1 报
`No matching distribution found for filelock`,紧接着
`No module named 'typing_extensions'` 和 `import torch 失败` —— 看着像 torch 坏了,
其实 torch 本体已经装上,缺的是它 import 时要的几个普通 PyPI 包。成因有三层:
torch 的 wheel 是 `download.pytorch.org` 的直链,不经索引,所以它下得动;
`pip install -U pip setuptools wheel` 在索引不通时也不报错(本来就装着,查不到
新版就算已满足);而补依赖那一行以前没有失败检查。现在脚本在装任何东西之前先探索引,
不通就停下并打出 pip 的原话、`pip config list` 和已设置的 proxy / `PIP_*` 变量名。
修法是换镜像(`PIP_IDX=<镜像地址> bash env/2_pip_train.sh`)或把 proxy 切回
上次 pip 装成功时的状态。

**`download.pytorch.org` 连不上长什么样。** pip 反复打
`Retrying ... ConnectTimeoutError ... download.pytorch.org`。它和 PyPI 镜像对 proxy
的要求可能正好相反 —— 一个通,另一个就超时,同一次运行里没法两全。做法是把 torch
那个 wheel 单独弄下来:

```bash
TORCH_WHL=<本地 .whl 的路径> bash env/2_pip_train.sh
```

或者先手动 `pip install --no-deps <本地 .whl>` 再直接跑脚本:环境里已经是核对过的
`2.14.0.dev20260719` 时,脚本会跳过下载,不再去连 `download.pytorch.org`
(torch_npu 同理)。手动装 torch_npu 时带上 `--no-deps`:它的 wheel 只声明了一条依赖
`torch==2.14.0.dev20260719`,环境里不是这个版本时 pip 会去索引里找这个 nightly,
找不到就整条失败。

两个安装脚本里 pip 都不带 `-q`:输出长,但哪一步、哪个包、连的哪个地址出问题
都直接看得到。

版本组合(2026-11-05 复查,两个 wheel 都仍可下载):

| 包 | 版本 |
|---|---|
| torch | `2.14.0.dev20260719+cpu` cp312 aarch64(直链) |
| torch_npu | `2.14.0.dev20260808` |
| triton-ascend | `3.2.1` |
| fla-core | `0.5.2`(+ `einops`;Qwen 模型专用) |
| torchvision | `0.29.0.dev20260720+cpu` cp312 aarch64(直链,`--no-deps`;它钉 `torch==2.14.0.dev20260719`)+ `pillow` |
| numpy | `>=2.0,<2.3` |
| attn-gym | `0.0.9` |
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
* **`import torch` 会顺带加载 torch_npu。** torch_npu 在 entry_points 里注册了
  `[torch.backends]`,只要它装着,`import torch` 就自动把它拉起来 —— 于是 torch_npu 的
  任何毛病都会被报成 `Failed to load the backend extension: torch_npu`,看着像 torch 坏了。
  想单独验 torch 本体:`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch'`
  (安装脚本的闸门 1 就是这么跑的)。
* **torch_npu 的 wheel 没声明它的运行依赖。** `import torch_npu` 要 numpy,源码顶层
  还 import yaml 和 packaging,但 METADATA 里只有 `torch==...` 一条(2026-11-16 读
  `2.14.0.dev20260808` 的 wheel 核过)。所以先手动装了 torch_npu、其余包还没装时,
  会得到 `No module named 'numpy'`。安装脚本现在在闸门 1 之前就把这三个装上。
* **numpy 必须落在 `>=2.0,<2.3`。** 上界来自 `scipy==1.13.1`(要求 `numpy<2.3`);
  下界来自 `datasets` 拉进来的新版 pyarrow —— 它在 import 时检查 numpy,1.x 直接报
  `pyarrow requires NumPy 2.0 or newer`,闸门 3 就过不去。坑在于 pyarrow 的元数据里
  **没声明** numpy,所以 pip 不会替你升:环境里先有了一个 numpy 1.26.4(手动装的、
  或内部镜像给的旧版),只要求 `numpy<2.3` 会被判为"已满足"原样留下。脚本现在两头都钉。
  本脚本装的包在 PyPI 上没有一个要求 `numpy<2`(2026-11-16 逐个查过)。
* **闸门 3/4 里的 `aclInit ... error code is 107001` / `Invalid device ID` 是预期的。**
  这两道闸门故意把 `ASCEND_RT_VISIBLE_DEVICES` 置空以免占卡,torchtitan_npu 的一个补丁
  想初始化 NPU 没成,自己打一行 `[PATCH] ... skipped` 跳过。失败原因只看 Traceback。
* **Qwen 模型有两个 requirements 里没列的硬依赖:`fla-core` 和 `triton-ascend`。**
  upstream torchtitan 的 `models/qwen3_5/model.py` 在模块顶层
  `from fla.modules.conv.triton.ops import ...`,而 torchtitan 和 torchtitan-npu 的
  requirements 都没列它,只有 `models/qwen3_5/README.md` 里一句
  `pip install av torchvision flash-linear-attention`。所以闸门 3(只 import 训练入口)
  能过,闸门 4(第一次 import Qwen 模型)才报 `No module named 'fla'`。
  `fla` 是 flash-linear-attention 的包名,算子在它的子包 `fla-core` 里,只装
  `fla-core==0.5.2 einops` 就够。钉 0.5.2 是因为它的 `[npu]` 配套正好是
  `triton-ascend==3.2.1`。fla 的算子又在顶层 import `triton`(由 triton-ascend 提供),
  所以 triton-ascend 也是闸门 4 的前提,不再是"训练前补上就行"。它的索引连不上时
  直接下 wheel(约 188 MB):
  `https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`
* **训练入口还需要 `torchvision`,闸门 4 验不到。** `torchtitan_npu.models.qwen3_5`
  一 import 就拉多模态数据管线(`hf_datasets/multimodal/utils/image.py` 顶层
  `import torchvision.io`),只训文本也绕不开。版本要和 torch 成对:
  `0.29.0.dev20260720`,nightly 索引页已经不列这个日期,但直链还在。
  安装脚本的闸门 5 就是为这类"闸门 4 的入口走不到"的缺口加的。
* **环境不要建在 `/dev/shm`。** 上一份训练环境就建在那里,随重启整个消失。
* **CANN 版本决定融合算子可用性。** 9.0.1 缺 `aclnnMhcPost` / `aclnnSparseFlashMla`
  等符号,9.1.0 才有;而且 `libopapi.so` 在 ops 包里,只装 toolkit 验不出来。
  `env/2_pip_train.sh` 的第 0 段会列出找到的每个 `libopapi.so` 和导出符号数,
  参照是 8.5.1=3511 / 9.0.1=8109 / 9.1.0=8835;不到 8000 会打警告。
  **8.x 上能不能跑通训练线没验过** —— 核对过的只有 9.1.0。
* **产物不要放 tmpfs。** 2bit 产物 7.5 GiB、3bit 11.3 GiB,放持久盘。
