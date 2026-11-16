"""让 DSv4 的分组专家 LoRA converter 能用在 Qwen 的配置树上 —— **不改 torchtitan-npu**。

为什么需要这个文件
------------------
`torchtitan_npu/models/deepseek_v4/lora.py` 在**已提交**的 `3f9f99a` 里,
`DeepSeekV4LoRAConverter.convert()` 的最后一步是::

    return derive(converted_root, _LoRAModelConfig, lora=...)

而 `_LoRAModelConfig` 是写死的 `class _LoRAModelConfig(DeepSeekV4Model.Config)`。
`derive` 按目标类的字段从源配置里抄值,目标类是 DSv4 的根配置,Qwen 的根配置
没有 `hc_head` / `compress_ratios` / `n_layers` / `window_size`,于是::

    TypeError: _LoRAModelConfig.__init__() missing 4 required keyword-only
    arguments: 'hc_head', 'compress_ratios', 'n_layers', and 'window_size'

注意这是 `convert()` 的**最后一行**。走到这里说明前面的树遍历、后缀匹配、
`grouped_lora_class` 工厂、`_validate_matches`(strict)全都过了 ——
分组专家的 LoRA 是挂得上 Qwen 的,只是最后给根配置加 `lora` 字段时用错了基类。

同事 `l84414662` 的工作区里这一处已经修了(`_lora_model_config_cls(type(root))`,
按实际根配置类生成),但**那个修复没有提交**。本项目的工单是照那个工作区核的,
所以在一份干净的 clone 上闸门 4 才第一次暴露出来(2026-11-16)。

做法
----
不去改 torchtitan-npu 的文件(干净 clone 改了就脏、同事的 clone 不许动)。
这里派生一个 converter,只在 `convert()` 期间把那个模块全局 `_LoRAModelConfig`
换成"以实际根配置类为基类"的版本,出来就还原。`derive(...)` 是在调用时才查这个
全局名的,所以这样换是有效的。

树里已经有 `_lora_model_config_cls`(= 已经带那个修复)时,本文件什么都不做,
直接返回原来的 converter。

用法::

    from qwen36_lora_compat import lora_converter_config
    conv = lora_converter_config(rank=16, alpha=32.0, rank_experts=16, target_modules=())
    spec = model_registry("35B-A3B", converters=[conv])
"""

# 刻意不写 `from __future__ import annotations`:下面动态生成的 dataclass 的字段
# 注解要是真的类对象,不要变成一个事后解析不出来的字符串。

from dataclasses import dataclass
from functools import cache

__all__ = ["lora_converter_config", "shim_active", "describe"]


def _dsv4_lora_module():
    from torchtitan_npu.models.deepseek_v4 import lora as mod

    return mod


def shim_active() -> bool:
    """True = torchtitan-npu 的 lora.py 是已提交的原版,需要本文件的兼容层。"""
    return not hasattr(_dsv4_lora_module(), "_lora_model_config_cls")


@cache
def _root_cls(base_cls: type, lora_cfg_cls: type) -> type:
    """以 `base_cls`(实际的模型根配置类)为基类、多一个 `lora` 字段的根配置类。

    @cache:同一个基类只生成一个类,`isinstance` / 重复转换才稳定。
    继承下来的 `_owner` 不变,所以 `build()` 构造的仍是原来的模型类。
    """

    @dataclass(kw_only=True, slots=True)
    class _LoRAModelConfig(base_cls):  # type: ignore[valid-type, misc]
        lora: lora_cfg_cls  # type: ignore[valid-type]

    _LoRAModelConfig.__name__ = f"LoRA{base_cls.__name__}"
    _LoRAModelConfig.__qualname__ = f"LoRA{base_cls.__qualname__}"
    return _LoRAModelConfig


@cache
def _converter_cls() -> type:
    mod = _dsv4_lora_module()
    base = mod.DeepSeekV4LoRAConverter
    if not shim_active():
        return base

    class Qwen36LoRAConverter(base):  # type: ignore[valid-type, misc]
        """`DeepSeekV4LoRAConverter`,但根配置按实际模型类派生。"""

        @dataclass(kw_only=True, slots=True)
        class Config(base.Config):  # type: ignore[name-defined, misc]
            pass

        def convert(self, model_config):
            prev = mod._LoRAModelConfig
            mod._LoRAModelConfig = _root_cls(type(model_config), type(self.config))
            try:
                return super().convert(model_config)
            finally:
                mod._LoRAModelConfig = prev

    return Qwen36LoRAConverter


def lora_converter_config(**kwargs):
    """`DeepSeekV4LoRAConverter.Config(**kwargs)` 的替身,字段完全一样。"""
    return _converter_cls().Config(**kwargs)


def describe() -> str:
    """一行话说明当前走的是哪条路,给 probe 和启动日志用。"""
    if shim_active():
        return ("torchtitan-npu 的 lora.py 是已提交的原版(根配置写死为 DSv4)"
                " -> 用本仓库 qwen36_lora_compat 的兼容层")
    return "torchtitan-npu 的 lora.py 已带 _lora_model_config_cls -> 直接用原 converter"
