#!/usr/bin/env bash
# 阶段 2(量化线):装包 + 跑自测。**这一步必须关掉 proxy。**
#
#     bash env/2_pip_quant.sh
#
# 前置:bash env/1_conda_envs.sh(那一步需要 proxy)。
#
# 脚本内部用 env -u 把 proxy 变量从每个联网命令上剥掉,所以即使你忘了
# unset,或者 proxy 是在 ~/.bashrc 里 export 的,这里也不会带上它。
set -uo pipefail

QUANT_ENV="${QUANT_ENV:-$HOME/conda-envs/lqq-quant}"
PIP_IDX="${PIP_IDX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$QUANT_ENV/bin/python"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$1"; shift; for l in "$@"; do printf '   %s\n' "$l"; done; exit 1; }

# 关键:把 proxy 从子进程环境里彻底去掉,而不是设成空串 —— 有些工具
# 对 "设了但为空" 和 "没设" 的处理不同。
no_proxy_run() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u all_proxy -u ALL_PROXY -u ftp_proxy -u FTP_PROXY "$@"
}

[ -x "$P" ] || die "找不到 $P" "先跑 bash env/1_conda_envs.sh(那一步需要 proxy)"

say "proxy 状态"
echo "当前 shell 里的 proxy 变量:"
env | grep -iE '^(http|https|all|ftp)_proxy=' || echo "  (无)"
echo "下面所有 pip 命令都会用 env -u 剥掉这些变量。"

say "装包  (python: $("$P" -V 2>&1))"
no_proxy_run "$P" -m pip install -q -U -i "$PIP_IDX" pip setuptools wheel \
    || no_proxy_run "$P" -m pip install -q -U pip setuptools wheel \
    || die "pip 自举失败" "镜像不可达就 PIP_IDX=https://pypi.org/simple 重跑"

no_proxy_run "$P" -m pip install -i "$PIP_IDX" -r "$REPO_ROOT/env/quant-requirements.txt" \
    || no_proxy_run "$P" -m pip install -r "$REPO_ROOT/env/quant-requirements.txt" \
    || die "装量化线依赖失败" \
           "镜像不可达就 PIP_IDX=https://pypi.org/simple 重跑" \
           "若是 proxy 仍在生效,检查 ~/.pip/pip.conf 或 ~/.config/pip/pip.conf 里有没有 proxy ="

say "版本确认"
no_proxy_run "$P" -c 'import torch, safetensors, numpy; print("torch", torch.__version__); print("safetensors", safetensors.__version__); print("numpy", numpy.__version__)' \
    || die "import 失败"

say "自测 1/2:量化内核  (24 项)"
no_proxy_run "$P" -I "$REPO_ROOT/verify/test_core_cpu.py" || die "内核自测未过"

say "自测 2/2:端到端假量化"
no_proxy_run "$P" -I "$REPO_ROOT/verify/test_fakequant_cpu.py" || die "端到端自测未过"

say "量化线就绪"
cat <<NOTE
解释器:$P

拿到真权重后第一步是看键名分类对不对(只读 header,不写文件):

    $P -I src/fakequant_rtn.py --in <ckpt> --scan

确认分类无歧义后再做 3bit 假量化:

    $P -I src/fakequant_rtn.py --in <ckpt> --out <dst> --bits 3 --group 128

注意 2bit 不要用 RTN(实测比 LDLQ+RHT 差 6.4 dB),要等 offline_ptq.py。
NOTE
