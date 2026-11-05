#!/usr/bin/env bash
# 阶段 2(训练线):装包 + 过四道闸门。**这一步必须关掉 proxy。**
# 工单阶段 6(LoRA 训练)才需要;量化线用不到。
#
#     bash env/2_pip_train.sh
#
# 前置:bash env/1_conda_envs.sh(那一步需要 proxy)。
#
# 四道闸门,每道失败都会打出下一步该查什么:
#     GATE 1  torch           装上且能 import
#     GATE 2  torch_npu       能 import(ABI 匹配 + CANN 能找到)—— 最容易炸
#     GATE 3  torchtitan_npu  能 import(torch 版本够新)
#     GATE 4  LoRA 配置树     分组专家 LoRA 能挂上 Qwen3.5-35B-A3B
#
# 全程不占 NPU:GATE 4 置空 ASCEND_RT_VISIBLE_DEVICES,纯 CPU 跑配置树。
# 真正上卡的 5 步 dry run 是脚本最后单独给出的一条命令。
set -uo pipefail

TRAIN_ENV="${TRAIN_ENV:-$HOME/conda-envs/lqq-train}"
TT_DIR="${TT_DIR:-$HOME/src/torchtitan-npu}"
PIP_IDX="${PIP_IDX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# git clone 默认也**不走** proxy(gitcode.com 是国内源)。要走就设 GIT_PROXY。
GIT_PROXY="${GIT_PROXY:-}"

# 2026-11-05 复查:两个 wheel 都仍可下载。**不要只升一个**,见 GATE 2。
TORCH_WHL="${TORCH_WHL:-https://download.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260719%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl}"
TORCH_NPU_VER="${TORCH_NPU_VER:-2.14.0.dev20260808}"
TORCH_NPU_IDX="${TORCH_NPU_IDX:-https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torch-npu/index.html}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$TRAIN_ENV/bin/python"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$1"; shift; for l in "$@"; do printf '   %s\n' "$l"; done; exit 1; }

no_proxy_run() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u all_proxy -u ALL_PROXY -u ftp_proxy -u FTP_PROXY "$@"
}
git_run() {
    if [ -n "$GIT_PROXY" ]; then
        http_proxy="$GIT_PROXY" https_proxy="$GIT_PROXY" git "$@"
    else
        no_proxy_run git "$@"
    fi
}

[ -x "$P" ] || die "找不到 $P" "先跑 bash env/1_conda_envs.sh(那一步需要 proxy)"

say "proxy 状态"
echo "当前 shell 里的 proxy 变量:"
env | grep -iE '^(http|https|all|ftp)_proxy=' || echo "  (无)"
echo "pip 与 git 都会用 env -u 剥掉这些变量(git 可用 GIT_PROXY 单独开)。"

# ---------------------------------------------------------------- CANN
say "0. CANN 环境"
SET_ENV=""
for c in "${ASCEND_SET_ENV_PATH:-}" \
         /usr/local/Ascend/ascend-toolkit/set_env.sh \
         /usr/local/Ascend/cann/set_env.sh; do
    [ -n "$c" ] && [ -f "$c" ] && SET_ENV="$c" && break
done
[ -n "$SET_ENV" ] || die "找不到 CANN 的 set_env.sh" \
    "找过 ascend-toolkit/set_env.sh 和 cann/set_env.sh" \
    "装在别处就 export ASCEND_SET_ENV_PATH=<路径> 再跑"
# shellcheck disable=SC1090
source "$SET_ENV"
echo "已 source $SET_ENV"
echo "toolkit: $(readlink -f /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null || echo '<未知>')"
LIB="$(find /usr/local/Ascend -name libopapi.so 2>/dev/null | head -1)"
if [ -n "$LIB" ]; then
    echo "libopapi: $LIB"
    echo "  符号总数: $(nm -D "$LIB" 2>/dev/null | grep -c ' T ')   (参考 9.0.1=8109 / 9.1.0=8835)"
    for s in aclnnMoeDistributeDispatch aclnnFlashAttentionScoreV3 aclnnFlashAttentionVarLenScoreV3; do
        n="$(nm -D "$LIB" 2>/dev/null | grep -c "$s")"
        printf '  %-40s %s%s\n' "$s" "$n" "$([ "$n" -eq 0 ] && echo '  <- Qwen 路径要用,为 0 要注意')"
    done
else
    echo "没找到 libopapi.so —— 它在 ops 包里(A3 芯片叫 Ascend-cann-A3-ops),不在 toolkit 里"
fi

say "pip 自举"
no_proxy_run "$P" -m pip install -q -U -i "$PIP_IDX" pip setuptools wheel \
    || no_proxy_run "$P" -m pip install -q -U pip setuptools wheel \
    || die "pip 自举失败" "镜像不可达就 PIP_IDX=https://pypi.org/simple 重跑"

# ---------------------------------------------------------------- GATE 1
say "1. torch  (GATE 1)"
echo "wheel: $TORCH_WHL"
no_proxy_run "$P" -m pip install -q --no-deps "$TORCH_WHL" \
    || die "torch 下载/安装失败" \
           "直链可能已被回收。去 https://download.pytorch.org/whl/nightly/cpu/torch/" \
           "找最接近 2026-07-19 的 aarch64 cp312 nightly,用 TORCH_WHL=<url> 覆盖。" \
           "**换日期有风险**,见 GATE 2 的说明。"
no_proxy_run "$P" -m pip install -q -i "$PIP_IDX" filelock fsspec jinja2 networkx sympy typing-extensions \
    || no_proxy_run "$P" -m pip install -q filelock fsspec jinja2 networkx sympy typing-extensions
no_proxy_run "$P" -c 'import torch; print("torch", torch.__version__)' || die "import torch 失败"

# ---------------------------------------------------------------- GATE 2
say "2. torch_npu $TORCH_NPU_VER  (GATE 2 —— 最容易炸的一步)"
no_proxy_run "$P" -m pip install -q --no-deps --find-links "$TORCH_NPU_IDX" "torch_npu==$TORCH_NPU_VER" \
    || die "torch_npu 安装失败" \
           "看索引里现有哪些版本:" \
           "  curl -sS $TORCH_NPU_IDX | grep -oE 'torch_npu-[0-9.a-z]*'" \
           "2026-11-05 时有 2.12.0.dev20260808 / 2.14.0.dev20260808 / 2.14.0.dev20260918"
if ! no_proxy_run "$P" -c 'import torch, torch_npu; print("torch_npu", torch_npu.__version__)'; then
    die "import torch_npu 失败" \
        "两种典型原因:" \
        "  a) ABI 不符:报 'Mismatch in kernel C++ signatures' 然后 abort。" \
        "     torch_npu 是照某个具体 torch nightly 编的 —— dev20260808 对的是 torch dev20260719。" \
        "     aten::_scaled_grouped_mm_v2 的 Tensor[] 参数在 0719 到 0731 之间从" \
        "     c10::ArrayRef 改成了 c10::IListRef,跨过去就炸。两个必须成对换。" \
        "  b) CANN 找不到:确认上面 source 的 set_env.sh 是对的那个,且" \
        "     LD_LIBRARY_PATH 没被 ~/.bashrc 里的绝对赋值覆盖掉。"
fi

# ---------------------------------------------------------------- deps
say "3. torchtitan-npu 与其余依赖"
if [ ! -d "$TT_DIR/.git" ]; then
    mkdir -p "$(dirname "$TT_DIR")"
    git_run clone https://gitcode.com/cann/torchtitan-npu.git "$TT_DIR" \
        || die "clone torchtitan-npu 失败" "要让 git 走 proxy 就 GIT_PROXY=http://<host>:<port> 重跑"
fi
git_run -C "$TT_DIR" submodule update --init --recursive \
    || die "submodule(third_party/torchtitan)拉取失败" "同上,可试 GIT_PROXY"
no_proxy_run "$P" -m pip install -q --no-deps -e "$TT_DIR/third_party/torchtitan" || die "装 torchtitan 失败"
no_proxy_run "$P" -m pip install -q --no-deps -e "$TT_DIR" || die "装 torchtitan_npu 失败"
# requirements.txt 没列但缺的那批。spmd_types 必须是 0.2.3(torchtitan 0.3.0 钉的)。
no_proxy_run "$P" -m pip install -q -i "$PIP_IDX" \
    attn-gym==0.0.9 pybind11 numpy PyYAML ninja scipy==1.13.1 safetensors==0.8.0 \
    "spmd_types==0.2.3" torchdata tensorboard tyro blobfile tiktoken tomli_w tabulate \
    "datasets==4.7.0" tokenizers sentencepiece psutil wandb torch_checkpointing \
    || die "其余依赖安装失败" "镜像不可达就 PIP_IDX=https://pypi.org/simple 重跑"
no_proxy_run "$P" -m pip install -q --no-deps -i https://triton-ascend.osinfra.cn/pypi/simple "triton-ascend==3.2.1" \
    || echo "   (triton-ascend 装失败。Qwen 的 gated_delta 走 Triton kernel,训练前要补上)"

# ---------------------------------------------------------------- GATE 3
say "4. torchtitan_npu 能 import 吗  (GATE 3)"
if ! no_proxy_run env ASCEND_RT_VISIBLE_DEVICES= "$P" -c 'import torchtitan_npu.train; print("TRAIN IMPORT OK")'; then
    die "import torchtitan_npu 失败" \
        "若报 cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'," \
        "说明 torch 太旧(需要 2.14 系列)。GATE 1 装对了就不该出现这条。"
fi

# ---------------------------------------------------------------- GATE 4
say "5. 分组专家 LoRA 能挂上 Qwen3.5-35B-A3B 吗  (GATE 4)"
echo "这是整条 LoRA 线的闸门。纯 CPU,不占卡。"
no_proxy_run env ASCEND_RT_VISIBLE_DEVICES= "$P" -I "$REPO_ROOT/verify/probe_lora_tree.py"
RC=$?

say "完成"
echo "环境:   $TRAIN_ENV"
echo "解释器: $P"
echo "仓库:   $TT_DIR"
[ "$RC" -eq 0 ] && echo "四道闸门全过。" \
    || echo "GATE 4 未过(rc=$RC)—— 上面的输出会指出是 import 问题还是判据不成立。"
cat <<NOTE

下一步(需要一张空闲卡):5 步 dry run,确认真权重能加载。
    cd $TT_DIR
    export ASCEND_SET_ENV_PATH=$SET_ENV
    export PATH=$TRAIN_ENV/bin:\$PATH
    NGPU=1 MODULE=torchtitan_npu.models.qwen3_5 CONFIG=qwen35_27b_4k_sft \\
      HF_ASSETS_PATH=<tokenizer 目录> CKPT_SAVE_LOAD_PATH=<一个空目录> \\
      bash scripts/run_train.sh --training.steps 5 --optimizer.name AdamW --compile.no-enable

注意 scripts/run_train.sh 直接调 torchrun。系统 PATH 上若有别的 torch,
要确保 $TRAIN_ENV/bin 在前面(日志里确认用的是本环境的 torchrun)。
NOTE
