#!/usr/bin/env bash
# triton-ascend 的编译器包装:让它在"torch 2.14 + CANN 8.x + GCC 10"上能编出 kernel 的启动代码。
#
# 不用手动设:src/qwen36_triton_env.py(recipe 导入时调)会把 CC 指到这个脚本。
# triton-ascend 取编译器时读的就是 CC(triton/backends/ascend/utils.py::_get_cxx)。
#
# triton-ascend 3.2.1 每个 kernel 第一次运行时要现编一小段 C++ 启动代码。
# 2026-11-17 第一次跑到 GDN kernel,在这套环境上三处编不过:
#
#   1. 它写死了 -std=c++17,而 torch 2.14 的 ATen.h 要求 C++20:
#        #error C++20 or later compatible compiler is required to use ATen.
#      torchtitan-npu 自己的测试里也把这一条标成了已知问题(xfail,
#      experiments/torchao-npu/tests/integration/test_training.py)。
#   2. 换成 -std=c++20 之后,GCC 10 的 __cplusplus 仍是 201709L(GCC 11 起才是
#      202002L),ATen.h 的检查还是过不去。头文件本身在 GCC 10 的 C++20 模式下
#      是编得过的(实测),卡住的只是那个版本号。
#   3. torch_npu 的头文件引用了 CANN 9.x 才有的类型,CANN 8.x 的头里没有:
#        'aclmdlRICondHandle' has not been declared
#
# 这个脚本不改任何第三方包,只对**带 -std=c++17 的那几条命令**动手:
#   * -std=c++17 换成 C++20;
#   * 编译器自报的 __cplusplus 不到 202002L 时,显式定义成 202002L;
#   * CANN 头里确实缺那两个类型时,用 -include 带上 triton_cann_compat.h 补声明。
# 其它命令(不带 -std=c++17 的)参数原样交给真正的编译器。
#
# 真正的编译器:QWEN36_REAL_CXX,没设就用 PATH 上的 g++。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="${QWEN36_REAL_CXX:-$(command -v g++ || true)}"
if [ -z "$REAL" ]; then
    echo "triton_cxx.sh: 找不到 g++(可用 QWEN36_REAL_CXX 指定编译器)" >&2
    exit 127
fi

# GCC 10 起才认 -std=c++20,8/9 要写 c++2a,更老的没有 C++20
STD="-std=c++20"
MAJOR="$("$REAL" -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
case "$MAJOR" in
    8|9) STD="-std=c++2a" ;;
    [1-7])
        echo "triton_cxx.sh: $REAL 是 GCC $MAJOR,没有 C++20。torch 2.14 的头文件需要 GCC >= 10(8/9 未验证)。" >&2
        echo "               装一个新的 g++,用 QWEN36_REAL_CXX=<路径> 指给本脚本。" >&2
        ;;
esac

args=()
replaced=0
acl_has_cond=""
for a in "$@"; do
    case "$a" in
        -std=c++17|-std=gnu++17)
            args+=("$STD")
            replaced=1
            ;;
        -I*)
            args+=("$a")
            d="${a#-I}"
            if [ -z "$acl_has_cond" ] && [ -f "$d/acl/acl.h" ]; then
                if grep -rqs "aclmdlRICondHandle" "$d/acl"; then acl_has_cond=yes; else acl_has_cond=no; fi
            fi
            ;;
        *) args+=("$a") ;;
    esac
done

if [ "$replaced" -ne 1 ]; then
    exec "$REAL" "$@"
fi

extra=()
ver="$(echo | "$REAL" "$STD" -x c++ -E -dM - 2>/dev/null | awk '$2 == "__cplusplus" {print $3}' | tr -d 'L')"
if [ -n "$ver" ] && [ "$ver" -lt 202002 ]; then
    extra+=(-U__cplusplus -D__cplusplus=202002L)
fi
if [ "$acl_has_cond" = "no" ]; then
    extra+=(-include "$HERE/triton_cann_compat.h")
fi

exec "$REAL" "${extra[@]}" "${args[@]}"
