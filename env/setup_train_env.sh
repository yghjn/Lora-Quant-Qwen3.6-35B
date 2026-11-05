#!/usr/bin/env bash
# LoRA 训练线的环境安装(工单阶段 6 才需要)。量化线用不到 —— 见 env/quant.yml。
#
#     bash env/setup_train_env.sh
#
# 版本组合是 2026-10 在 Atlas 800 A3 上验证通过的那一组,2026-11-05 复查
# 两个 wheel 仍可下载。**不要随意升级** —— 理由见 GATE 2 的说明。
#
# 脚本走到底会过四道闸门,每道失败都会打出下一步该查什么:
#     GATE 1  torch        装上且能 import
#     GATE 2  torch_npu    能 import(ABI 匹配 + CANN 能找到)
#     GATE 3  torchtitan_npu  能 import(torch 版本够新)
#     GATE 4  LoRA 配置树  分组专家 LoRA 能挂上 Qwen3.5-35B-A3B
#
# 全程不占 NPU:GATE 4 置空 ASCEND_RT_VISIBLE_DEVICES,纯 CPU 跑配置树。
# 真正上卡的 5 步 dry run 是最后单独一条命令,留给卡空闲时手动跑。
set -uo pipefail

ENV_NAME="${ENV_NAME:-lqq-train}"
PY_VER="${PY_VER:-3.12}"
# **不要放 /dev/shm** —— 上一份环境建在那里,随重启整个消失。
PREFIX="${PREFIX:-$HOME/conda-envs/$ENV_NAME}"
# torchtitan-npu 的落地位置。它不在本仓库里(是上游 + 华为的 BSD-3 项目)。
TT_DIR="${TT_DIR:-$HOME/src/torchtitan-npu}"
PIP_IDX="${PIP_IDX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

TORCH_WHL="${TORCH_WHL:-https://download.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260719%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl}"
TORCH_NPU_VER="${TORCH_NPU_VER:-2.14.0.dev20260808}"
TORCH_NPU_IDX="${TORCH_NPU_IDX:-https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torch-npu/index.html}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$PREFIX/bin/python"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() {
    printf '\n\033[1;31m!! %s\033[0m\n' "$1"
    shift
    for l in "$@"; do printf '   %s\n' "$l"; done
    exit 1
}

# ---------------------------------------------------------------- CANN
say "0. CANN 环境"
SET_ENV=""
for c in /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/cann/set_env.sh \
         "${ASCEND_SET_ENV_PATH:-}"; do
    [ -n "$c" ] && [ -f "$c" ] && SET_ENV="$c" && break
done
if [ -z "$SET_ENV" ]; then
    die "找不到 CANN 的 set_env.sh" \
        "找过:/usr/local/Ascend/ascend-toolkit/set_env.sh 和 cann/set_env.sh" \
        "装在别处就 export ASCEND_SET_ENV_PATH=<路径> 再跑本脚本"
fi
# shellcheck disable=SC1090
source "$SET_ENV"
echo "已 source $SET_ENV"
TK="$(readlink -f /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null || true)"
echo "toolkit: ${TK:-<未知>}"
LIBOPAPI="$(find /usr/local/Ascend -name libopapi.so 2>/dev/null | head -1)"
if [ -n "$LIBOPAPI" ]; then
    echo "libopapi: $LIBOPAPI"
    echo "  符号总数: $(nm -D "$LIBOPAPI" 2>/dev/null | grep -c ' T ')  (参考 9.0.1=8109 / 9.1.0=8835)"
    for s in aclnnMoeDistributeDispatch aclnnFlashAttentionScoreV3; do
        n="$(nm -D "$LIBOPAPI" 2>/dev/null | grep -c "$s")"
        printf '  %-34s %s%s\n' "$s" "$n" "$([ "$n" -eq 0 ] && echo '   <- Qwen 路径要用,为 0 要注意')"
    done
else
    echo "没找到 libopapi.so —— 它在 ops 包里(A3 芯片叫 Ascend-cann-A3-ops),不在 toolkit 里"
fi

# ---------------------------------------------------------------- env
say "1. conda 环境 $ENV_NAME (python $PY_VER) -> $PREFIX"
if [ -x "$P" ]; then
    echo "已存在,跳过创建。要重建就先 rm -rf $PREFIX"
else
    conda create -y -p "$PREFIX" "python=$PY_VER" || die "conda create 失败"
fi
"$P" -m pip install -q -U -i "$PIP_IDX" pip setuptools wheel \
    || "$P" -m pip install -q -U pip setuptools wheel \
    || die "pip 自举失败" "镜像不可达就 export PIP_IDX=https://pypi.org/simple"

# ---------------------------------------------------------------- GATE 1
say "2. torch (GATE 1)"
echo "wheel: $TORCH_WHL"
"$P" -m pip install -q --no-deps "$TORCH_WHL" \
    || die "torch 下载/安装失败" \
           "直链可能已被回收。去 https://download.pytorch.org/whl/nightly/cpu/torch/" \
           "找最接近 2026-07-19 的 aarch64 cp312 nightly,用 TORCH_WHL=<url> 覆盖。" \
           "**换日期有风险**,见 GATE 2 的说明。"
"$P" -m pip install -q -i "$PIP_IDX" filelock fsspec jinja2 networkx sympy typing-extensions \
    || "$P" -m pip install -q filelock fsspec jinja2 networkx sympy typing-extensions
"$P" -c 'import torch; print("torch", torch.__version__)' || die "import torch 失败"

# ---------------------------------------------------------------- GATE 2
say "3. torch_npu $TORCH_NPU_VER (GATE 2 —— 最容易炸的一步)"
"$P" -m pip install -q --no-deps --find-links "$TORCH_NPU_IDX" "torch_npu==$TORCH_NPU_VER" \
    || die "torch_npu 安装失败" \
           "看索引里现有哪些版本:curl -sS $TORCH_NPU_IDX | grep -oE 'torch_npu-[0-9.a-z]*'" \
           "2026-11-05 时有:2.12.0.dev20260808 / 2.14.0.dev20260808 / 2.14.0.dev20260918"
if ! "$P" -c 'import torch, torch_npu; print("torch_npu", torch_npu.__version__)'; then
    die "import torch_npu 失败" \
        "两种典型原因:" \
        "  a) ABI 不符:报 'Mismatch in kernel C++ signatures' 然后 abort。" \
        "     torch_npu 是照某个具体 torch nightly 编的 —— dev20260808 这版对的是" \
        "     torch dev20260719。aten::_scaled_grouped_mm_v2 的 Tensor[] 参数在" \
        "     0719 到 0731 之间从 c10::ArrayRef 改成了 c10::IListRef,跨过去就炸。" \
        "     所以 torch 和 torch_npu 必须成对换,不要只升一个。" \
        "  b) CANN 找不到:确认上面 source 的 set_env.sh 是对的那个," \
        "     且 LD_LIBRARY_PATH 没被 .bashrc 里的绝对赋值覆盖掉。"
fi

# ---------------------------------------------------------------- deps
say "4. torchtitan-npu 与其余依赖"
if [ ! -d "$TT_DIR/.git" ]; then
    mkdir -p "$(dirname "$TT_DIR")"
    git clone https://gitcode.com/cann/torchtitan-npu.git "$TT_DIR" || die "clone torchtitan-npu 失败"
fi
git -C "$TT_DIR" submodule update --init --recursive || die "submodule(third_party/torchtitan)拉取失败"
"$P" -m pip install -q --no-deps -e "$TT_DIR/third_party/torchtitan" || die "装 torchtitan 失败"
"$P" -m pip install -q --no-deps -e "$TT_DIR" || die "装 torchtitan_npu 失败"
# requirements.txt 没列但缺的那批。spmd_types 必须是 0.2.3(torchtitan 0.3.0 钉的)。
"$P" -m pip install -q -i "$PIP_IDX" \
    attn-gym==0.0.9 pybind11 numpy PyYAML ninja scipy==1.13.1 safetensors==0.8.0 \
    "spmd_types==0.2.3" torchdata tensorboard tyro blobfile tiktoken tomli_w tabulate \
    "datasets==4.7.0" tokenizers sentencepiece psutil wandb torch_checkpointing \
    || die "其余依赖安装失败" "镜像不可达就 export PIP_IDX=https://pypi.org/simple 重跑"
"$P" -m pip install -q --no-deps -i https://triton-ascend.osinfra.cn/pypi/simple "triton-ascend==3.2.1" \
    || echo "   (triton-ascend 装失败。Qwen 的 gated_delta 走 Triton kernel,训练前要补上)"

# ---------------------------------------------------------------- GATE 3
say "5. torchtitan_npu 能 import 吗 (GATE 3)"
if ! ASCEND_RT_VISIBLE_DEVICES= "$P" -c 'import torchtitan_npu.train; print("TRAIN IMPORT OK")'; then
    die "import torchtitan_npu 失败" \
        "若报 cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'," \
        "说明 torch 太旧(需要 2.14 系列)。GATE 1 装对了就不该出现这条。"
fi

# ---------------------------------------------------------------- GATE 4
say "6. 分组专家 LoRA 能挂上 Qwen3.5-35B-A3B 吗 (GATE 4)"
echo "这是整条 LoRA 线的闸门。纯 CPU,不占卡。"
ASCEND_RT_VISIBLE_DEVICES= "$P" -I "$REPO_ROOT/verify/probe_lora_tree.py"
RC=$?

say "完成"
echo "环境:    $PREFIX"
echo "解释器:  $P"
echo "仓库:    $TT_DIR"
if [ "$RC" -eq 0 ]; then
    echo "四道闸门全过。"
else
    echo "GATE 4 未过(rc=$RC)—— 上面的输出会指出是 import 问题还是判据不成立。"
fi
cat <<NOTE

下一步(需要一张空闲卡):5 步 dry run,确认真权重能加载。
    cd $TT_DIR
    export ASCEND_SET_ENV_PATH=$SET_ENV
    export PATH=$PREFIX/bin:\$PATH
    NGPU=1 MODULE=torchtitan_npu.models.qwen3_5 CONFIG=qwen35_27b_4k_sft \\
      HF_ASSETS_PATH=<tokenizer 目录> CKPT_SAVE_LOAD_PATH=<一个空目录> \\
      bash scripts/run_train.sh --training.steps 5 --optimizer.name AdamW --compile.no-enable

注意 scripts/run_train.sh 直接调 torchrun。系统 PATH 上若有别的 torch,
要确保 $PREFIX/bin 在前面(日志里确认用的是本环境的 torchrun)。
NOTE
