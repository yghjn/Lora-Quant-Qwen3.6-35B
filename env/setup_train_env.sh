#!/usr/bin/env bash
# LoRA 训练线的环境(工单阶段 6 才需要)。量化线用不到这个 —— 见 env/quant.yml。
#
# 为什么单独一份:torchtitan-npu 把 torch 钉死在一个 nightly 日期上,
# 和量化线想要的"任意较新 torch"不兼容,必须隔离。
#
#     bash env/setup_train_env.sh
#
# 本脚本**刻意不自动执行到底**:带 [CHECK] 的地方需要按目标机的实际情况确认。
# 版本号来自 2026-10 在一台 Atlas 800 A3 / CANN 9.1.0 上验证过的安装记录,
# 换机器必须重新核对,照抄会在 ABI 上炸。
set -euo pipefail

ENV_NAME="${ENV_NAME:-lqq-train}"
PY_VER="${PY_VER:-3.12}"
# **不要放 /dev/shm** —— 上一份环境就是建在那里,随重启整个消失。
PREFIX="${PREFIX:-$HOME/conda-envs/$ENV_NAME}"

echo "=== 1. 建 conda 环境 $ENV_NAME (python $PY_VER) ==="
conda create -y -p "$PREFIX" "python=$PY_VER"
P="$PREFIX/bin/python"
TUNA="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
$P -m pip install -U -i "$TUNA" pip setuptools wheel

cat <<'NOTE'

=== 2. [CHECK] torch 与 torch_npu ===
这一步是整套环境里唯一容易炸的地方,所以不自动跑。

上一份验证过的组合(Atlas 800 A3 / CANN 9.1.0 / 驱动 25.5.1):
    torch      2.14.0.dev20260719+cpu   (aarch64, cp312)
    torch_npu  2.14.0.dev20260808

torch 的日期**必须精确**。nightly 索引页已不列 20260719,但直链当时仍可用:
  https://download.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260719%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl

为什么不能用更新的:`aten::_scaled_grouped_mm_v2` 的 Tensor[] 参数在
0719 到 0731 之间从 c10::ArrayRef 改成了 c10::IListRef,而 torch_npu 是照
0719 编的 —— 换成 0808 会在 `import torch_npu` 时直接
  "Mismatch in kernel C++ signatures" 然后 abort。

装法(两个都要 --no-deps):
  $P -m pip install --no-deps "<上面那个 torch 直链>"
  $P -m pip install -i $TUNA filelock fsspec jinja2 networkx sympy typing-extensions
  $P -m pip install --no-deps \
     --find-links https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torch-npu/index.html \
     "torch_npu==2.14.0.dev20260808"

**换机器前先确认三件事:**
  a) 目标机的 CANN 版本(cat /usr/local/Ascend/ascend-toolkit/latest/version.info)
     torchtitan-npu 的融合算子需要 CANN 里有 aclnnMhcPost / aclnnSparseFlashMla
     等符号;9.0.1 没有,9.1.0 才有。只装 toolkit 验不出来 —— libopapi.so 在
     ops 包里(A3 芯片的包名是 Ascend-cann-A3-ops)。
  b) 上面两个 wheel 的 URL 是否还在。
  c) aarch64 / cp312 是否与目标机匹配。

NOTE

echo "=== 3. [CHECK] torchtitan 与 torchtitan-npu ==="
cat <<'NOTE'
这两个不在本仓库里(是上游 + 华为的 BSD-3 项目),要自己取:

    git clone https://gitcode.com/cann/torchtitan-npu.git
    cd torchtitan-npu && git checkout master
    git submodule update --init --recursive        # third_party/torchtitan
    $P -m pip install --no-deps -e third_party/torchtitan
    $P -m pip install --no-deps -e .

**注意:** 本项目的 LoRA 线依赖 torchtitan-npu 的两样东西:
  * torchtitan_npu/models/deepseek_v4/lora.py 的 grouped_lora_class 工厂
    (已验证它对任何用上游通用 GroupedExperts 的 MoE 都成立,不绑 DeepSeek)
  * torchtitan_npu/models/qwen3_5/ 的 model_registry
上游 master 的 qwen3_5 **没有** 35B-A3B 的 trainer 配置,也没有 LoRA 配置,
那是本项目要加的(工单 §6.1)。
NOTE

echo "=== 4. 其余依赖 ==="
cat <<'NOTE'
  $P -m pip install -i $TUNA attn-gym==0.0.9 pybind11 numpy PyYAML ninja \
      scipy==1.13.1 safetensors==0.8.0 "spmd_types==0.2.3" torchdata tensorboard \
      tyro blobfile tiktoken tomli_w tabulate "datasets==4.7.0" tokenizers \
      sentencepiece psutil wandb torch_checkpointing
  $P -m pip install --no-deps -i https://triton-ascend.osinfra.cn/pypi/simple "triton-ascend==3.2.1"

block-FP8 量化训练(本项目默认不用)还需要:
  $P -m pip install "torchao==0.17.0" && $P -m pip install -e experiments/torchao-npu
NOTE

echo "=== 5. 验收 ==="
cat <<NOTE
  $P -c 'import torchtitan_npu.train; print("TRAIN IMPORT OK")'
  UNISVQ_NBITS=3 $P -I verify/probe_lora_tree.py    # 工单 0.1

probe_lora_tree.py 是本仓库里唯一需要训练环境的验证脚本。
它在 torch 2.10 上会卡在
    cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'
那正是上面要 torch 2.14 的原因。
NOTE

echo
echo "环境骨架已建在 $PREFIX"
echo "带 [CHECK] 的步骤请按目标机实际情况逐条确认后手工执行。"
