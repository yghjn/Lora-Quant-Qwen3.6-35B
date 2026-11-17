"""让 triton-ascend 用本仓库的编译器包装(``env/triton_cxx.sh``)编 kernel 的启动代码。

为什么需要
----------
Qwen3.5/3.6 有 30 层是 GatedDeltaNet,NPU 上它的 kernel 是 Triton-Ascend 写的
(``torchtitan_npu/ops/triton/gdn``)。triton-ascend 3.2.1 每个 kernel 第一次运行时要
现编一小段 C++ 启动代码,在"torch 2.14 + CANN 8.x + GCC 10"上编不过(2026-11-17
第一次前向时报的)::

    #error C++20 or later compatible compiler is required to use ATen.
    'aclmdlRICondHandle' has not been declared

原因和处理写在 ``env/triton_cxx.sh`` 的开头。triton-ascend 取编译器时读环境变量
``CC``(``triton/backends/ascend/utils.py::_get_cxx``),所以把 ``CC`` 指到那个包装
脚本就行,不用改任何第三方包。

这个模块只做一件事:在**还没人设 CC** 的时候把它设上。recipe 导入时调一次 ——
每个 rank 的进程都会导入 recipe,而 kernel 是在第一次前向时才编的,来得及。

``CC`` 已经被设成别的东西时不覆盖(那是使用者自己的选择),只打一条告警说明
GDN kernel 可能编不过、以及怎么把两者接起来。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["wrapper_path", "ensure_triton_compiler"]


def wrapper_path() -> str:
    """``<仓库>/env/triton_cxx.sh`` 的绝对路径。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "env", "triton_cxx.sh")


def ensure_triton_compiler() -> str:
    """把 ``CC`` 指到包装脚本(没人设过时)。返回一句话说明做了什么,给日志和探针用。"""
    wrapper = wrapper_path()
    current = os.environ.get("CC")

    if current and os.path.abspath(current) == wrapper:
        return f"CC 已是 {wrapper}"

    if current:
        msg = (
            f"CC 已被设成 {current!r},不覆盖。triton-ascend 会用它编 kernel 的启动代码,"
            "在 torch 2.14 + CANN 8.x 上可能编不过。要两者都用:"
            f"QWEN36_REAL_CXX={current} CC={wrapper}"
        )
        logger.warning(msg)
        return msg

    if not os.path.isfile(wrapper):
        msg = f"找不到 {wrapper} —— CC 没设,triton-ascend 会直接用 g++"
        logger.warning(msg)
        return msg
    if not os.access(wrapper, os.X_OK):
        # git 会保留可执行位;走到这里多半是文件被拷贝/解包过
        try:
            os.chmod(wrapper, os.stat(wrapper).st_mode | 0o111)
        except OSError as exc:
            msg = f"{wrapper} 没有可执行权限且改不了({exc})—— 手动 chmod +x 后重跑"
            logger.warning(msg)
            return msg

    os.environ["CC"] = wrapper
    return f"CC -> {wrapper}"
