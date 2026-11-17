#!/usr/bin/env python3
"""Triton-Ascend 的 kernel 启动代码在这台机器上编得出来吗。

    python3 verify/probe_triton_launcher_cpu.py
    python3 verify/probe_triton_launcher_cpu.py --raw     # 不用本仓库的编译器包装,看原始报错

纯 CPU、不占 NPU,约半分钟。只编译、不执行 kernel。

为什么要它
----------
Qwen3.6 有 30 层 GatedDeltaNet,NPU 上的 kernel 是 Triton-Ascend 写的。triton-ascend
每个 kernel **第一次运行时**才现编一小段 C++ 启动代码 —— 也就是说,编不过要等到
8 张卡加载完 70 GB 权重、跑到第一次前向才知道(2026-11-17 就是这样发现的)::

    #error C++20 or later compatible compiler is required to use ATen.
    'aclmdlRICondHandle' has not been declared

这个脚本把 triton-ascend 自己的那两步原样调一遍:

1. 预编译头(``utils._precompile_npu_ext``);
2. 编启动代码的 .so(``utils._build_npu_ext``),再 import 它,确认有 ``launch``。

第 2 步用的启动代码是 triton-ascend 自己的生成器(``driver.generate_npu_wrapper_src``)
按一个假想 kernel 的签名生成的 —— 和真 kernel 的启动代码是同一个模板,include 的
头文件、链接的库都一样。

编译器走的是 recipe 会设的那个 ``CC``(``src/qwen36_triton_env.py`` ->
``env/triton_cxx.sh``)。``--raw`` 把它去掉,用来确认这个探针拦得住原始的报错。

没验到的:kernel 本身在卡上能不能跑(那要 NPU)。

退出码:0 通过;1 编不过;2 环境问题(缺模块等)。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import types

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _first_errors(exc: BaseException, limit: int = 8) -> list[str]:
    msg = str(exc)
    cut = msg.find(",cmd=")
    body = msg[:cut] if cut > 0 else msg
    lines = [line.strip() for line in body.splitlines() if "error" in line.lower()]
    return [line[:200] for line in lines[:limit]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", action="store_true", help="不用本仓库的编译器包装(自检:应当编不过)")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        from triton.backends.ascend import driver, utils
    except ModuleNotFoundError as exc:
        print(f"!! 缺模块 {exc.name} —— 先让 env/2_pip_train.sh 前面的闸门过了。")
        return 2

    if args.raw:
        os.environ.pop("CC", None)
        print("编译器: 不带包装(--raw),triton-ascend 自己挑的 " + str(utils._get_cxx()))
    else:
        from qwen36_triton_env import ensure_triton_compiler

        print("编译器: " + ensure_triton_compiler())
    cxx = os.environ.get("QWEN36_REAL_CXX") or "g++"
    try:
        ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        ver = "(取不到版本)"
    print(f"         {cxx}: {ver}")
    try:
        print(f"CANN:    {utils._get_ascend_path()}")
    except Exception as exc:  # noqa: BLE001
        print(f"CANN:    (取不到路径: {exc})")

    ok = True

    def step(name: str, fn):
        nonlocal ok
        began = time.time()
        try:
            detail = fn()
            print(f"  OK  {name}: {detail}({time.time() - began:.0f} 秒)")
            return True
        except BaseException as exc:  # noqa: BLE001
            ok = False
            print(f"  X   {name}: {type(exc).__name__}")
            for line in _first_errors(exc) or [str(exc)[:300]]:
                print("        " + line)
            return False

    with tempfile.TemporaryDirectory(prefix="qwen36_triton_probe_") as tmp:
        header = os.path.join(tmp, "precompiled.h")
        with open(header, "w", encoding="utf-8") as fh:
            fh.write(driver.generate_npu_header_src())

        def precompile():
            utils._precompile_npu_ext(header, header + ".gch")
            return f"{os.path.getsize(header + '.gch') / 1e6:.0f} MB"

        print("\n================ 判据 ================")
        if step("1. 预编译头(ATen.h + torch_npu 头 + CANN 头)", precompile):
            # 一个假想 kernel 的元数据,字段取 generate_npu_wrapper_src 实际读的那些
            meta = types.SimpleNamespace(
                debug=False, mix_mode="aiv", shared=False, compile_on_910_95=False,
                parallel_mode="", force_simt_only=False, shared_mem_dynamic_size=0,
                name="probe_kernel", kernel_name="probe_kernel",
                workspace_size=64, lock_num=1, lock_init_value=0, bs_task_type=0,
            )
            src_path = os.path.join(tmp, "launcher_probe.cxx")
            built: dict[str, str] = {}

            def build():
                wrapper_src = driver.generate_npu_wrapper_src({}, {0: "*fp32", 1: "*fp32", 2: "i32"}, meta)
                with open(src_path, "w", encoding="utf-8") as fh:
                    fh.write(wrapper_src)
                built["so"] = utils._build_npu_ext(
                    "launcher_probe", header, src_path, kernel_launcher="torch", precompile=True
                )
                return f"{os.path.getsize(built['so']) / 1e3:.0f} KB"

            def load():
                # 模块名必须是这个:生成的启动代码里导出的是 PyInit___triton_launcher,
                # triton-ascend 自己(driver.py 的 launcher)也是按这个名字 import 的。
                spec = importlib.util.spec_from_file_location("__triton_launcher", built["so"])
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "launch"):
                    raise RuntimeError("编出来的模块里没有 launch")
                return "有 launch 入口"

            if step("2. 编启动代码的 .so", build):
                step("3. .so 能 import(符号都解析得到)", load)

    if ok:
        print("\n结论:通过。")
        return 0
    print("\n结论:未通过(退出码 1)。")
    if args.raw:
        print("(--raw 模式下编不过是预期的:这就是不带包装时训练第一次前向会报的错。)")
    else:
        print("包装脚本是 env/triton_cxx.sh,开头写了它处理的三种情况。上面的 error 行不在其中的话,")
        print("多半是编译器太老(需要 GCC >= 10)或 CANN 头文件和 torch_npu 差得更远。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
