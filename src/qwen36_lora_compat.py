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

__all__ = ["lora_converter_config", "ensure_adapter_sharding", "shim_active", "describe", "ADAPTER_PARAMS"]


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


ADAPTER_PARAMS = ("w13_lora_a", "w1_lora_b", "w3_lora_b", "w2_lora_a", "w2_lora_b")


def ensure_adapter_sharding() -> list[str]:
    """让 Qwen 的专家切分声明认得 5 个 adapter 参数。返回这次新加的名字(可重复调用)。

    **为什么需要**(2026-11-16,第一次上卡跑到 ``model.parallelize`` 才暴露)::

        ValueError: LoRA_ClampGroupedExperts.w13_lora_a has no placement declared
        in sharding_config.state_shardings.

    ``Module._distribute_states``(upstream ``protocols/module.py``)要求模块自己的
    **每一个**参数都在 ``sharding_config.state_shardings`` 里有声明。这份声明是
    ``set_moe_sharding_config(..., expert_param_layout=...)`` 按一张"参数名 -> TP 上
    怎么切"的表生成的,而这张表每个模型各有一份:

    * DSv4 的(``torchtitan_npu/models/deepseek_v4/sharding.py``)把 5 个 adapter
      和 3 个基座权重**列在一起**,所以 DSv4 的 LoRA 能并行化;
    * Qwen3.5 的(upstream ``torchtitan/models/qwen3_5/sharding.py``)只有
      ``w1_EFD`` / ``w2_EDF`` / ``w3_EFD``。converter 把模块换成了带 adapter 的,
      声明表却还是原来那张。

    做法:把 DSv4 表里 Qwen 表没有的 adapter 条目**原样抄过去**(不自己写一份,
    以 DSv4 那张为准)。抄的是模块全局的那个 dict 对象,``_set_qwen35_layer_sharding``
    调用时按名字取它,所以对之后所有的 ``update_from_config`` 生效。

    对不带 LoRA 的 Qwen 配置无影响:声明表里多出来的名字没有对应参数,不会被用到
    (DSv4 不开 LoRA 时用的就是那张带 adapter 名字的表)。

    开了 EP 时(我们的配置是 EP=8),表里的 TP 切法其实不参与:所有专家参数统一按
    专家维(dim 0)在 EP 轴上切。5 个 adapter 的 dim 0 都是专家数,和基座权重一致。
    """
    from torchtitan.models.qwen3_5 import sharding as qwen_sharding

    from torchtitan_npu.models.deepseek_v4 import sharding as dsv4_sharding

    name = "_GROUPED_EXPERTS_PARAM_LAYOUT"
    qwen_layout = getattr(qwen_sharding, name, None)
    dsv4_layout = getattr(dsv4_sharding, name, None)
    if not isinstance(qwen_layout, dict) or not isinstance(dsv4_layout, dict):
        raise RuntimeError(
            f"找不到 {name}(qwen3_5: {type(qwen_layout).__name__},"
            f" deepseek_v4: {type(dsv4_layout).__name__})—— 上游的切分声明换了写法,"
            "qwen36_lora_compat.ensure_adapter_sharding 要跟着改。"
        )
    missing = [p for p in ADAPTER_PARAMS if p not in dsv4_layout]
    if missing:
        raise RuntimeError(f"DSv4 的切分声明里没有 adapter 参数 {missing} —— 参数名变了?")

    added = [p for p in ADAPTER_PARAMS if p not in qwen_layout]
    for p in added:
        qwen_layout[p] = dsv4_layout[p]
    return added


def lora_converter_config(**kwargs):
    """`DeepSeekV4LoRAConverter.Config(**kwargs)` 的替身,字段完全一样。

    顺带保证 adapter 的切分声明已经补上 —— 用了这个 converter 就一定需要它。
    """
    ensure_adapter_sharding()
    return _converter_cls().Config(**kwargs)


def describe() -> str:
    """一行话说明当前走的是哪条路,给 probe 和启动日志用。"""
    if shim_active():
        return ("torchtitan-npu 的 lora.py 是已提交的原版(根配置写死为 DSv4)"
                " -> 用本仓库 qwen36_lora_compat 的兼容层")
    return "torchtitan-npu 的 lora.py 已带 _lora_model_config_cls -> 直接用原 converter"
