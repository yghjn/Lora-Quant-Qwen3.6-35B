#!/usr/bin/env bash
# 阶段 1:**只建 conda 环境,一个包都不装。** 这一步需要 proxy。
#
#     CONDA_PROXY=http://<host>:<port> bash env/1_conda_envs.sh
# 或者(proxy 已在 shell 里 export 过 / 写在 ~/.condarc 里):
#     bash env/1_conda_envs.sh
#
# 装包在阶段 2,那里必须**关掉** proxy:
#     bash env/2_pip_quant.sh      # 量化线
#     bash env/2_pip_train.sh      # 训练线(工单阶段 6 才需要)
#
# 为什么拆开:本机的约束是 conda 走 proxy、pip 不能走 proxy。
# `conda env create` 如果 yml 里带 pip: 段,会在 conda 操作内部调 pip,
# 那时 proxy 还开着 —— 两个要求没法在同一个进程里同时满足。
# 所以 env/quant.yml 刻意只写 python 和 pip 本身。
set -uo pipefail

QUANT_ENV="${QUANT_ENV:-$HOME/conda-envs/lqq-quant}"
# **不要放 /dev/shm** —— 上一份环境建在那里,随重启整个消失。
TRAIN_ENV="${TRAIN_ENV:-$HOME/conda-envs/lqq-train}"
TRAIN_PY="${TRAIN_PY:-3.12}"   # torchtitan-npu 的 wheel 是 cp312
QUANT_PY="${QUANT_PY:-3.10}"
WANT_TRAIN="${WANT_TRAIN:-1}"  # 设 0 则只建量化线的环境

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*"; exit 1; }

# conda 走 proxy。CONDA_PROXY 没设就按"调用者自己配好了"处理(shell 里
# export 过,或写在 ~/.condarc 的 proxy_servers 里),不替它猜。
run_conda() {
    if [ -n "${CONDA_PROXY:-}" ]; then
        http_proxy="$CONDA_PROXY" https_proxy="$CONDA_PROXY" \
        HTTP_PROXY="$CONDA_PROXY" HTTPS_PROXY="$CONDA_PROXY" \
        conda "$@"
    else
        conda "$@"
    fi
}

command -v conda >/dev/null || die "找不到 conda"

say "proxy 状态"
if [ -n "${CONDA_PROXY:-}" ]; then
    echo "conda 将使用 CONDA_PROXY=$CONDA_PROXY"
else
    echo "CONDA_PROXY 未设 —— 按当前 shell / ~/.condarc 的配置走。"
    echo "当前 shell 里的 proxy 变量:"
    env | grep -iE '^(http|https|all)_proxy=' || echo "  (无)"
    echo "conda 装不上就设 CONDA_PROXY=http://<host>:<port> 重跑。"
fi

say "量化线环境 -> $QUANT_ENV  (python $QUANT_PY)"
if [ -x "$QUANT_ENV/bin/python" ]; then
    echo "已存在,跳过。要重建先 rm -rf $QUANT_ENV"
else
    run_conda create -y -p "$QUANT_ENV" "python=$QUANT_PY" \
        || die "建量化线环境失败。若是网络问题,设 CONDA_PROXY 重跑。"
fi
"$QUANT_ENV/bin/python" -V

if [ "$WANT_TRAIN" = "1" ]; then
    say "训练线环境 -> $TRAIN_ENV  (python $TRAIN_PY)"
    if [ -x "$TRAIN_ENV/bin/python" ]; then
        echo "已存在,跳过。要重建先 rm -rf $TRAIN_ENV"
    else
        run_conda create -y -p "$TRAIN_ENV" "python=$TRAIN_PY" \
            || die "建训练线环境失败。只想做量化线就 WANT_TRAIN=0 重跑。"
    fi
    "$TRAIN_ENV/bin/python" -V
else
    say "跳过训练线环境(WANT_TRAIN=0)"
fi

say "阶段 1 完成"
cat <<NOTE
两个环境都是空的,刻意没装任何包。

接下来**把 proxy 关掉**再跑阶段 2:

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    bash env/2_pip_quant.sh        # 量化线,装完会直接跑两个自测
    bash env/2_pip_train.sh        # 训练线,装完会过四道闸门

阶段 2 的脚本内部也会用 env -u 再剥一遍 proxy 变量,所以忘了 unset 也没关系。
NOTE
