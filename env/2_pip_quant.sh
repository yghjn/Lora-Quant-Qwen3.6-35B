#!/usr/bin/env bash
# 阶段 2(量化线):装包 + 跑自测。
#
#     conda activate lqq-quant
#     bash env/2_pip_quant.sh
#
# proxy 由你自己开关 —— 本脚本不碰任何 proxy 变量。
# 本机的情况是:**这一步(pip)需要 proxy 关掉。**
#
# 幂等:包已经装过就直接跑自测,不会重复下载。
set -uo pipefail

PIP_IDX="${PIP_IDX:-}"   # 留空用 pip 自己的默认源;要换镜像就设它
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$1"; shift; for l in "$@"; do printf '   %s\n' "$l"; done; exit 1; }

# 认当前激活的环境。没激活就退回 QUANT_ENV 指定的路径。
if [ -n "${CONDA_PREFIX:-}" ]; then
    P="$CONDA_PREFIX/bin/python"
elif [ -n "${QUANT_ENV:-}" ] && [ -x "$QUANT_ENV/bin/python" ]; then
    P="$QUANT_ENV/bin/python"
else
    die "没有激活的 conda 环境" \
        "先 conda activate lqq-quant 再跑本脚本," \
        "或者 QUANT_ENV=<环境目录> bash env/2_pip_quant.sh"
fi
[ -x "$P" ] || die "$P 不可执行"

pip_install() {
    if [ -n "$PIP_IDX" ]; then "$P" -m pip install -i "$PIP_IDX" "$@"; else "$P" -m pip install "$@"; fi
}

say "环境"
echo "解释器: $P"
"$P" -V

say "装包"
pip_install -U pip setuptools wheel || die "pip 自举失败" "网络问题请确认 proxy 已按本机要求关掉"
pip_install -r "$REPO_ROOT/env/quant-requirements.txt" \
    || die "装量化线依赖失败" \
           "网络问题请确认 proxy 已按本机要求关掉" \
           "要换镜像:PIP_IDX=https://pypi.tuna.tsinghua.edu.cn/simple bash env/2_pip_quant.sh"

say "版本"
"$P" -c 'import torch, safetensors, numpy
print("torch       ", torch.__version__)
print("safetensors ", safetensors.__version__)
print("numpy       ", numpy.__version__)' || die "import 失败"

say "自测 1/2:量化内核  (24 项)"
"$P" -I "$REPO_ROOT/verify/test_core_cpu.py" || die "内核自测未过"

say "自测 2/2:端到端假量化"
"$P" -I "$REPO_ROOT/verify/test_fakequant_cpu.py" || die "端到端自测未过"

say "量化线就绪"
cat <<NOTE
解释器:$P

拿到真权重后第一步是看键名分类对不对(只读 header,不写文件):

    python -I src/fakequant_rtn.py --in <ckpt> --scan

确认分类无歧义后再做 3bit 假量化:

    python -I src/fakequant_rtn.py --in <ckpt> --out <dst> --bits 3 --group 128

2bit 不要用 RTN(实测比 LDLQ+RHT 差 6.4 dB),要等 offline_ptq.py。
NOTE
