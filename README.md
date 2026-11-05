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
python -I src/fakequant_rtn.py --in /path/to/Qwen3.6-35B-A3B \
    --out /path/to/qwen36-3bit-rtn --bits 3 --group 128
```

产出的是一份 **bf16 的 checkpoint**(量化再反量化),所以任何普通 bf16 推理或
训练路径都能直接加载,不需要低位宽 kernel。位宽是 Ŵ 的信息量属性,不是容器属性。

## 量化范围:只做路由专家

和 DeepSeek-V4.1 的现行做法一致(已核实:量化 runtime 与训练侧加载器里
`shared_expert` 一次都没出现,产物按路由专家编号组织,重建只写 FusedMoE 的
`w13_weight` / `w2_weight`)。

| 部分 | 参数量 | 占比 | 处理 |
|---|---:|---:|---|
| 路由专家 | **32.21 G** | **92%** | **量化到 2/3 bit** |
| embeddings + lm_head | 1.02 G | 3% | 原精度 |
| attention | 约 0.84 G | 2.4% | 原精度 |
| 共享专家 | 0.13 G | 0.4% | 原精度 |

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
verify/
  test_core_cpu.py         内核自测,24 项,纯 CPU
  test_fakequant_cpu.py    端到端自测,合成 checkpoint
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
| §2.2 `offline_ptq.py` | 未做 |
| §6 LoRA 训练 | 未做 |

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

### 两个会改排期的发现

**一、RTN 在 3bit 可当脚手架,在 2bit 必须放弃。**

| bits | RTN(group 128) | LDLQ + RHT | 差距 |
|---:|---:|---:|---:|
| 2 | 2.31 dB | 8.67 dB | **6.4 dB** |
| 3 | 11.26 dB | 12.04 dB | 0.8 dB |

3bit 下只差 0.8 dB,**完全够打通整条回路**;2bit 下差 6.4 dB,
所以 2bit 必须等 Hessian 路线(`offline_ptq.py`),不能用 RTN 交付。
`fakequant_rtn.py` 在 `--bits 2` 时会打这条警告。

**二、纯 CPU 做全量 PTQ 约 8 小时,不是工单里估的 2–4 小时。**
按实测单矩阵耗时,每层 256 专家约 12 分钟,40 层约 8 小时(单线程)。
工单那个估计隐含了用 NPU 做 `ldlq` 里的矩阵乘,而那个 Python 循环
(n=2048 时 512 次迭代)不随设备变快,所以上卡也压不到 1 小时以下。
**排期按卡的可用性分两档写。**

## 下一步

与卡的可用性解耦,按这个顺序:

1. **不需要卡**:下 Qwen3.6-35B-A3B 权重并逐文件校验(工单 §0.2);
   `--scan` 核对键名分类;用真 config 重跑形状预检(§0.3)。
2. **不需要卡**:按 `env/setup_train_env.sh` 建训练环境,
   跑 `verify/probe_lora_tree.py`(§0.1)。这是整条 LoRA 线的闸门。
3. **不需要卡**:`fakequant_rtn.py --bits 3` 产出 bf16 checkpoint(§1.1)。
4. 需要卡:bf16 基线评测、`offline_ptq.py`、LoRA 训练、合并复评。

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
