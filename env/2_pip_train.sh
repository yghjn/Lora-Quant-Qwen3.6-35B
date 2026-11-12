#!/usr/bin/env bash
# 阶段 2(训练线):装包 + 过四道闸门。工单阶段 6(LoRA 训练)才需要。
#
#     conda activate lqq-train
#     bash env/2_pip_train.sh
#
# proxy 由你自己开关 —— 本脚本不碰任何 proxy 变量。
# 本机的情况是:**这一步(pip / git)需要 proxy 关掉。**
#
# 四道闸门,每道失败都会打出下一步该查什么:
#     GATE 1  torch           装上且能 import
#     GATE 2  torch_npu       能 import(ABI 匹配 + CANN 能找到)—— 最容易炸
#     GATE 3  torchtitan_npu  能 import(torch 版本够新)
#     GATE 4  LoRA 配置树     分组专家 LoRA 能挂上 Qwen3.5-35B-A3B
#
# 全程不占 NPU:GATE 3/4 置空 ASCEND_RT_VISIBLE_DEVICES,纯 CPU。
# 真正上卡的 5 步 dry run 是脚本最后单独给出的一条命令。
set -uo pipefail

TT_DIR="${TT_DIR:-$HOME/src/torchtitan-npu}"
PIP_IDX="${PIP_IDX:-}"   # 留空用 pip 自己的默认源

# 2026-11-05 复查:两个 wheel 都仍可下载。**不要只升一个**,见 GATE 2。
TORCH_WHL_USER="${TORCH_WHL:-}"   # 非空 = 用户自己换了 wheel,版本/架构检查放行
TORCH_WHL="${TORCH_WHL:-https://download.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260719%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl}"
TORCH_NPU_VER="${TORCH_NPU_VER:-2.14.0.dev20260808}"
TORCH_NPU_IDX="${TORCH_NPU_IDX:-https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torch-npu/index.html}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$1"; shift; for l in "$@"; do printf '   %s\n' "$l"; done; exit 1; }

if [ -n "${CONDA_PREFIX:-}" ]; then
    P="$CONDA_PREFIX/bin/python"
elif [ -n "${TRAIN_ENV:-}" ] && [ -x "$TRAIN_ENV/bin/python" ]; then
    P="$TRAIN_ENV/bin/python"
else
    die "没有激活的 conda 环境" \
        "先 conda activate lqq-train 再跑本脚本," \
        "或者 TRAIN_ENV=<环境目录> bash env/2_pip_train.sh"
fi
[ -x "$P" ] || die "$P 不可执行"

pip_install() {
    if [ -n "$PIP_IDX" ]; then "$P" -m pip install -i "$PIP_IDX" "$@"; else "$P" -m pip install "$@"; fi
}
cpu_only() { env ASCEND_RT_VISIBLE_DEVICES= "$@"; }

say "环境"
echo "解释器: $P"
"$P" -V
# Python 版本不对就在这里停,不要往下走。以前这里只打一行"注意"然后继续,
# 结果是 CANN 段、pip 自举都跑完了才死在 GATE 1,而且 GATE 1 的报错说的是
# "直链可能已被回收" —— 和真因(解释器 tag 不符)毫无关系。
PYV="$("$P" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
PYMINOR="${PYV#*.}"
ENV_NAME="$(basename "${CONDA_PREFIX:-${TRAIN_ENV:-}}")"
if [ "${PYV%%.*}" != "3" ] || [ "$PYMINOR" -lt 12 ]; then
    WRONG_ENV=""
    case "$ENV_NAME" in
        *quant*) WRONG_ENV="当前环境 '$ENV_NAME' 是量化线的(Python 3.10,给 env/2_pip_quant.sh 用)。" ;;
    esac
    die "Python $PYV 装不了训练线,需要 3.12" \
        ${WRONG_ENV:+"$WRONG_ENV"} \
        "torchtitan-npu 的 requires-python 是 >=3.12,torch / torch_npu 的 wheel 也只有 cp312。" \
        "换 TORCH_WHL 救不了:就算 torch 装上,torchtitan_npu 也装不进 <3.12 的解释器。" \
        "" \
        "训练线用单独的环境 lqq-train:" \
        "    conda env list                      # 看 lqq-train 在不在" \
        "    bash env/1_conda_envs.sh            # 不在就建(已有的环境会跳过,不会动 lqq-quant)" \
        "    conda activate lqq-train && bash env/2_pip_train.sh"
fi
if [ "$PYV" != "3.12" ] && [ -z "$TORCH_WHL_USER" ]; then
    die "Python $PYV 不是 3.12,默认的 torch wheel 是 cp312 的,装不上" \
        "要么用 3.12 的环境(推荐,这是核对过的组合)," \
        "要么 TORCH_WHL=<对应 cp 版本的 nightly 直链> 重跑,并自己确认 torch_npu 有同 tag 的 wheel。"
fi
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ -z "$TORCH_WHL_USER" ]; then
    die "本机架构是 $ARCH,默认的 torch wheel 是 aarch64 的" \
        "Ascend 的训练环境都是 aarch64。确认你登的是带 NPU 的那台机器。"
fi

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
# set_env.sh 不是按 `set -u` 写的:调用方没设过 PYTHONPATH / CMAKE_PREFIX_PATH
# 时它会报两行 "unbound variable"。在 CANN 8.5.1 上实测那两行出在命令替换的
# 子 shell 里,最终环境和不开 -u 时一样,只是噪音;但别的版本的 set_env.sh
# 怎么写的没看过,所以 source 期间关掉 -u,不赌它。
set +u
# shellcheck disable=SC1090
source "$SET_ENV"
set -u
echo "已 source $SET_ENV"
echo "toolkit: $(readlink -f /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null || echo '<未知>')"
# 把找得到的 libopapi.so 全列出来(按真实路径去重)。以前只取 find 的第一个,
# 机器上并存几套 CANN 时看到的未必是 set_env.sh 实际指向的那套。
# 搜索范围:/usr/local/Ascend,加上 set_env.sh 所在的那棵树(CANN 装在别处时)。
CANN_ROOTS="/usr/local/Ascend"
case "$SET_ENV" in /usr/local/Ascend/*) ;; *) CANN_ROOTS="$CANN_ROOTS $(dirname "$SET_ENV")" ;; esac
# shellcheck disable=SC2086
LIBS="$(find $CANN_ROOTS -name libopapi.so 2>/dev/null | xargs -r -n1 readlink -f | sort -u)"
OPAPI_MAX=0
if [ -n "$LIBS" ]; then
    echo "libopapi.so(参考符号总数:8.5.1=3511 / 9.0.1=8109 / 9.1.0=8835):"
    for LIB in $LIBS; do
        tot="$(nm -D "$LIB" 2>/dev/null | grep -c ' T ')"
        [ "$tot" -gt "$OPAPI_MAX" ] && OPAPI_MAX="$tot"
        echo "  $LIB"
        echo "    符号总数: $tot"
        for s in aclnnMoeDistributeDispatch aclnnFlashAttentionScoreV3 aclnnFlashAttentionVarLenScoreV3; do
            n="$(nm -D "$LIB" 2>/dev/null | grep -c "$s")"
            printf '    %-40s %s%s\n' "$s" "$n" "$([ "$n" -eq 0 ] && echo '  <- Qwen 路径要用,为 0 要注意')"
        done
    done
    if [ "$OPAPI_MAX" -lt 8000 ]; then
        echo
        echo "  !! 这套 CANN 是 8.x 量级的(符号数不到 9.0.1 的一半)。"
        echo "     核对过能跑通 torchtitan-npu 训练的是 CANN 9.1.0。8.x 上 torch_npu"
        echo "     dev20260808 能不能 import、算子全不全,本项目没验过。"
        echo "     这不拦安装 —— 脚本继续,GATE 2(import torch_npu)会给出第一个实测答案。"
        echo "     机器上若另有 9.x:export ASCEND_SET_ENV_PATH=<那套的 set_env.sh> 再重跑。"
    fi
else
    echo "没找到 libopapi.so —— 它在 ops 包里(A3 芯片叫 Ascend-cann-A3-ops),不在 toolkit 里"
fi

# ---------------------------------------------------------------- 索引
say "pip 索引取得到包吗"
# 为什么单独探一次:索引不通时,下面的"pip 自举"**不会报错** —— pip / setuptools /
# wheel 本来就装着,-U 查不到新版本就算"已满足"。GATE 1 的 torch 走的是
# download.pytorch.org 直链,也不经索引。于是索引问题会一路滑到 torch 的运行依赖
# 那一行才露头,表现成 "No matching distribution found for filelock" 外加一个
# 看着像 torch 坏了的 "No module named 'typing_extensions'"。
# 这里拿一个小包(filelock,约 20 KB)真下载一次,不装进环境。
PROBE_DIR="$(mktemp -d)"
PROBE_LOG="$PROBE_DIR/pip.log"
PROBE_ARGS=(-m pip download -v --no-deps --no-cache-dir --retries 1 --timeout 20 -d "$PROBE_DIR" filelock)
[ -n "$PIP_IDX" ] && PROBE_ARGS+=(-i "$PIP_IDX")
mask() { sed -E 's#//[^/@[:space:]]*@#//<凭据已隐去>@#g'; }
if "$P" "${PROBE_ARGS[@]}" >"$PROBE_LOG" 2>&1; then
    echo "通。用的索引:$(grep -m1 -oE 'https?://[^ ]*/filelock/' "$PROBE_LOG" | mask || true)"
    rm -rf "$PROBE_DIR"
else
    echo "pip 的原话(筛过,凭据已隐去):"
    grep -E 'Looking in indexes|^ *\* http|HTTP error|Could not fetch|Retrying|ProxyError|SSLError|[Cc]onnection|name resolution|Skipping link|^ERROR' "$PROBE_LOG" \
        | mask | cut -c1-240 | head -14 | sed 's/^/    /'
    echo "pip config list:"
    CFG="$("$P" -m pip config list 2>/dev/null | mask)"
    printf '%s\n' "${CFG:-(空 —— 没有 pip.conf,用的是 pip 默认的 pypi.org)}" | sed 's/^/    /'
    echo "已设置的相关环境变量(只列名字,不打值):"
    VARS="$(env | grep -iE '^(pip_[a-z_]*|[a-z_]*proxy)=' | cut -d= -f1 | tr '\n' ' ')"
    echo "    ${VARS:-(无)}"
    rm -rf "$PROBE_DIR"
    die "pip 索引取不到包(用 filelock 探的),后面所有 pip 步骤都会失败" \
        "对照上面 pip 的原话:" \
        "  HTTP error 403/404/407,或没有任何网络报错只说 versions: none" \
        "      = 连上了,但那个地址给的不是包列表(镜像没同步 / 被网关或代理拦了)" \
        "  Retrying / Connection / name resolution" \
        "      = 根本连不上" \
        "两条修法,选一条:" \
        "  a) 换一个本机能通的镜像重跑,例如" \
        "       PIP_IDX=https://pypi.tuna.tsinghua.edu.cn/simple bash env/2_pip_train.sh" \
        "       PIP_IDX=https://repo.huaweicloud.com/repository/pypi/simple bash env/2_pip_train.sh" \
        "  b) proxy 状态不对。本脚本不碰 proxy,由你开/关后重跑 ——" \
        "     这台机器上次 pip 装成功(比如装 lqq-quant)时是什么状态,就切回那个状态。" \
        "注意:torch 的 wheel 是直链,它下得动不代表索引通。"
fi

say "pip 自举"
pip_install -q -U pip setuptools wheel || die "pip 自举失败" "网络问题请确认 proxy 已按本机要求关掉"

# ---------------------------------------------------------------- GATE 1
say "1. torch  (GATE 1)"
echo "wheel: $TORCH_WHL"
"$P" -m pip install -q --no-deps "$TORCH_WHL" \
    || die "torch 下载/安装失败" \
           "先看上面 pip 的原话:" \
           "  'is not a supported wheel on this platform' = wheel 的 tag 和解释器/架构不符," \
           "     不是链接问题(用了自定义 TORCH_WHL 时检查它的 cpXY 和 aarch64)。" \
           "  404 / 连接超时 = 直链可能已被回收,或 proxy 状态不对。" \
           "直链失效时去 https://download.pytorch.org/whl/nightly/cpu/torch/" \
           "找最接近 2026-07-19 的 aarch64 cp312 nightly,用 TORCH_WHL=<url> 覆盖。" \
           "**换日期有风险**,见 GATE 2 的说明。"
# torch 是 --no-deps 装的,它 import 时要的这几个纯 Python 包得单独补。
# 以前这行没有 `|| die`:装失败了脚本照样往下走,最后报的是 "import torch 失败"。
pip_install -q filelock fsspec jinja2 networkx sympy typing-extensions \
    || die "torch 的运行依赖安装失败(torch 本体已装上)" \
           "这几个都是 PyPI 上的普通包,装不上就是索引问题,不是 torch 的问题。" \
           "换镜像:PIP_IDX=https://pypi.tuna.tsinghua.edu.cn/simple 重跑"
"$P" -c 'import torch; print("torch", torch.__version__)' || die "import torch 失败"

# ---------------------------------------------------------------- GATE 2
say "2. torch_npu $TORCH_NPU_VER  (GATE 2 —— 最容易炸的一步)"
"$P" -m pip install -q --no-deps --find-links "$TORCH_NPU_IDX" "torch_npu==$TORCH_NPU_VER" \
    || die "torch_npu 安装失败" \
           "看索引里现有哪些版本:" \
           "  curl -sS $TORCH_NPU_IDX | grep -oE 'torch_npu-[0-9.a-z]*'" \
           "2026-11-05 时有 2.12.0.dev20260808 / 2.14.0.dev20260808 / 2.14.0.dev20260918"
if ! "$P" -c 'import torch, torch_npu; print("torch_npu", torch_npu.__version__)'; then
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
    git clone https://gitcode.com/cann/torchtitan-npu.git "$TT_DIR" || die "clone torchtitan-npu 失败"
fi
# upstream torchtitan **不是 submodule**:`.gitmodules` 里只有
# third_party/agent-skills/KPBot 一条,而 third_party/ 本身还被 torchtitan-npu
# 自己的 .gitignore 挡掉(第 115 行 `/third_party/`)。所以 clone 下来
# third_party/ 是空的,`git submodule update --init` 退出码 0 但什么都没拉到
# —— 本脚本之前就是这么写的,结果下一行必然 die 在"装 torchtitan 失败",
# 而那条消息完全看不出真因。改成显式 clone。
TT_UP="$TT_DIR/third_party/torchtitan"
if [ ! -d "$TT_UP/.git" ]; then
    mkdir -p "$(dirname "$TT_UP")"
    git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git "$TT_UP" \
        || die "clone upstream torchtitan 失败" \
               "它在 $TT_UP,不是 submodule,必须单独 clone。" \
               "手动做同一件事:git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git $TT_UP"
fi
# 钉在本项目核对过的那个 commit(torchtitan 0.3.0)。漂移了再往上走要重新核
# state_dict_adapter 的键映射和 model_registry 的签名。
git -C "$TT_UP" checkout -q "${TT_PIN:-086bf6c16}" 2>/dev/null \
    || echo "   (checkout ${TT_PIN:-086bf6c16} 没成功,用的是当前 HEAD:$(git -C "$TT_UP" log --oneline -1))"
# 其余 submodule(KPBot 那条)该拉还是拉,但失败不致命 —— 和本流程无关。
git -C "$TT_DIR" submodule update --init --recursive >/dev/null 2>&1 \
    || echo "   (submodule 拉取失败,与 torchtitan 无关,忽略)"
"$P" -m pip install -q --no-deps -e "$TT_UP" || die "装 torchtitan 失败" \
    "目录是 $TT_UP" \
    "--no-deps 是故意的:torchtitan 钉 torch==2.14.0.dev20260719,让 pip 去解会拆环境。"
"$P" -m pip install -q --no-deps -e "$TT_DIR" || die "装 torchtitan_npu 失败"
# requirements.txt 没列但缺的那批。spmd_types 必须是 0.2.3(torchtitan 0.3.0 钉的)。
pip_install -q \
    attn-gym==0.0.9 pybind11 numpy PyYAML ninja scipy==1.13.1 safetensors==0.8.0 \
    "spmd_types==0.2.3" torchdata tensorboard tyro blobfile tiktoken tomli_w tabulate \
    "datasets==4.7.0" tokenizers sentencepiece psutil wandb torch_checkpointing \
    || die "其余依赖安装失败" "要换镜像:PIP_IDX=https://pypi.tuna.tsinghua.edu.cn/simple 重跑"
"$P" -m pip install -q --no-deps -i https://triton-ascend.osinfra.cn/pypi/simple "triton-ascend==3.2.1" \
    || echo "   (triton-ascend 装失败。Qwen 的 gated_delta 走 Triton kernel,训练前要补上)"

# ---------------------------------------------------------------- GATE 3
say "4. torchtitan_npu 能 import 吗  (GATE 3)"
cpu_only "$P" -c 'import torchtitan_npu.train; print("TRAIN IMPORT OK")' \
    || die "import torchtitan_npu 失败" \
           "若报 cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'," \
           "说明 torch 太旧(需要 2.14 系列)。GATE 1 装对了就不该出现这条。"

# ---------------------------------------------------------------- GATE 4
say "5. 分组专家 LoRA 能挂上 Qwen3.5-35B-A3B 吗  (GATE 4)"
echo "这是整条 LoRA 线的闸门。纯 CPU,不占卡。"
# 这里**不要**加 `-I`。-I 隐含 -E(忽略 PYTHONPATH)和 -s(去掉 user site),
# 于是 GATE 4 失败后想用 PYTHONPATH / TT_ROOT 救一把的那条路也被堵死。
# 包是装进本 conda 环境的 prefix,隔离在这儿什么也不买。
cpu_only env TT_ROOT="$TT_DIR" "$P" "$REPO_ROOT/verify/probe_lora_tree.py"
RC=$?

say "完成"
echo "解释器: $P"
echo "仓库:   $TT_DIR"
[ "$RC" -eq 0 ] && echo "四道闸门全过。" \
    || echo "GATE 4 未过(rc=$RC)—— 上面的输出会指出是 import 问题还是判据不成立。"
cat <<NOTE

下一步(需要一张空闲卡):5 步 dry run,确认真权重能加载。
    cd $TT_DIR
    export ASCEND_SET_ENV_PATH=$SET_ENV
    NGPU=1 MODULE=torchtitan_npu.models.qwen3_5 CONFIG=qwen35_27b_4k_sft \\
      HF_ASSETS_PATH=<tokenizer 目录> CKPT_SAVE_LOAD_PATH=<一个空目录> \\
      bash scripts/run_train.sh --training.steps 5 --optimizer.name AdamW --compile.no-enable

注意 scripts/run_train.sh 直接调 torchrun。确认 PATH 上第一个 torchrun 来自
本环境($(dirname "$P")),否则会用到别的 torch。
NOTE
