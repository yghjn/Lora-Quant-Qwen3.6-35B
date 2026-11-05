#!/usr/bin/env bash
# 阶段 1:**只建 conda 环境,一个包都不装。**
#
#     bash env/1_conda_envs.sh
#
# proxy 由你自己开关 —— 本脚本不碰任何 proxy 变量。
# 本机的情况是:**这一步(conda)需要 proxy 开着。**
#
# 装包在阶段 2,那一步需要 proxy 关掉:
#     bash env/2_pip_quant.sh      # 量化线
#     bash env/2_pip_train.sh      # 训练线(工单阶段 6 才需要)
#
# 为什么拆开:conda 和 pip 对 proxy 的要求相反,没法在同一个进程里同时满足。
# 所以 env/quant.yml 刻意只写 python 和 pip 本身,不写 pip: 段 ——
# 否则 `conda env create` 会在 conda 操作内部调 pip,那时 proxy 还开着。
set -uo pipefail

QUANT_NAME="${QUANT_NAME:-lqq-quant}"
TRAIN_NAME="${TRAIN_NAME:-lqq-train}"
TRAIN_PY="${TRAIN_PY:-3.12}"   # torchtitan-npu 的 wheel 是 cp312
QUANT_PY="${QUANT_PY:-3.10}"
WANT_TRAIN="${WANT_TRAIN:-1}"  # 设 0 则只建量化线的环境

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*"; exit 1; }

command -v conda >/dev/null || die "找不到 conda。先 source <conda 安装目录>/etc/profile.d/conda.sh"

has_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

say "量化线环境 $QUANT_NAME  (python $QUANT_PY)"
if has_env "$QUANT_NAME"; then
    echo "已存在,跳过。要重建先 conda env remove -n $QUANT_NAME"
else
    conda create -y -n "$QUANT_NAME" "python=$QUANT_PY" \
        || die "建 $QUANT_NAME 失败。若是网络问题,确认 proxy 开着。"
fi

if [ "$WANT_TRAIN" = "1" ]; then
    say "训练线环境 $TRAIN_NAME  (python $TRAIN_PY)"
    if has_env "$TRAIN_NAME"; then
        echo "已存在,跳过。要重建先 conda env remove -n $TRAIN_NAME"
    else
        conda create -y -n "$TRAIN_NAME" "python=$TRAIN_PY" \
            || die "建 $TRAIN_NAME 失败。只想做量化线就 WANT_TRAIN=0 重跑。"
    fi
else
    say "跳过训练线环境(WANT_TRAIN=0)"
fi

say "阶段 1 完成"
cat <<NOTE
两个环境都是空的,刻意没装任何包。

接下来**把 proxy 关掉**,激活对应环境再跑阶段 2:

    conda activate $QUANT_NAME && bash env/2_pip_quant.sh
    conda activate $TRAIN_NAME && bash env/2_pip_train.sh

阶段 2 的脚本认当前激活的环境(读 CONDA_PREFIX),所以激活哪个就装进哪个。
NOTE
