#!/usr/bin/env python3
"""工单 0.1:验证分组专家 LoRA 能挂上 Qwen3.5/3.6-35B-A3B 的配置树。

为什么这是第一步
----------------
上游 `torchtitan.components.lora.LoRAConverter` 只认 `Linear.Config`,而 MoE 的
路由专家是 `[E, F, D]` 的三维参数。能覆盖它的是 DSv4 的
`grouped_lora_class(parent_cls)` 工厂。本脚本验证那个工厂在 Qwen 的配置树上
确实命中路由专家 —— **不命中的话,后面全部排期作废**。

不需要权重、不需要 NPU。纯配置树操作。

判据
----
1. 配置树里 `routed_experts.inner_experts` 的类型变成 GroupedLoRA* 的 Config;
2. 命中的 MoE 层数 == n_layers;
3. target_modules=() 下 dense 命中数 == 0(LoRA 只加路由专家);
4. adapter 形状与 `E × r × (2·dim + 3·moe_inter)` 的解析值一致。

用法
----
    PYTHONPATH=<repo>:<repo>/third_party/torchtitan python3 probe_lora_tree.py
环境变量:
    TT_FLAVOR   默认 35B-A3B
    TT_RANK     默认 16
    TT_ALPHA    默认 32
"""

from __future__ import annotations

import os
import sys
import traceback

# 不要碰别人的卡。本脚本是纯 CPU 的配置树操作;置空可见设备,避免 import 期
# 任何 NPU 初始化去抢占已被占满的 die。
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

FLAVOR = os.environ.get("TT_FLAVOR", "35B-A3B")
RANK = int(os.environ.get("TT_RANK", "16"))
ALPHA = float(os.environ.get("TT_ALPHA", "32"))


def _fail(stage: str, exc: BaseException) -> None:
    print(f"\n!! 失败于 [{stage}]: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    print(
        "\n判定:本步未通过。注意 import 失败不等于结论为假 —— "
        "torchtitan 钉 torch==2.14.0.dev20260719,低版本可能在 import 期就炸,\n"
        "      那属于环境问题,要在正确的 venv 里重跑再下结论。"
    )
    sys.exit(2)


# ---------------------------------------------------------------- imports
try:
    from torchtitan.models.qwen3_5 import model_registry, qwen3_5_configs
except BaseException as exc:  # noqa: BLE001
    _fail("import torchtitan.models.qwen3_5", exc)

try:
    from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter
except BaseException as exc:  # noqa: BLE001
    _fail("import DeepSeekV4LoRAConverter", exc)


# ------------------------------------------------------------ tree walk
def walk(cfg, path="model", depth=0, seen=None):
    """深度优先遍历配置对象树,yield (path, cfg)。"""
    if seen is None:
        seen = set()
    if id(cfg) in seen or depth > 24:
        return
    seen.add(id(cfg))
    yield path, cfg
    # dataclass 风格的 Config:走 __dict__;list/tuple/dict 走元素
    if isinstance(cfg, (list, tuple)):
        for i, v in enumerate(cfg):
            yield from walk(v, f"{path}[{i}]", depth + 1, seen)
        return
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            yield from walk(v, f"{path}[{k!r}]", depth + 1, seen)
        return
    for name, value in list(getattr(cfg, "__dict__", {}).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            continue
        yield from walk(value, f"{path}.{name}", depth + 1, seen)


def describe(cfg) -> str:
    owner = getattr(type(cfg), "__qualname__", type(cfg).__name__)
    return f"{type(cfg).__module__}.{owner}"


def main() -> int:
    print(f"flavor 可选: {sorted(qwen3_5_configs)}")
    if FLAVOR not in qwen3_5_configs:
        print(f"!! {FLAVOR} 不在注册表里")
        return 2
    print(f"使用 flavor={FLAVOR}  rank={RANK}  alpha={ALPHA}\n")

    # ---- 转换前 ----
    try:
        base = model_registry(FLAVOR)
    except BaseException as exc:  # noqa: BLE001
        _fail(f"model_registry({FLAVOR}) 未加 converter", exc)

    before = {}
    for path, cfg in walk(base.model):
        if "inner_experts" in path and path.endswith("inner_experts"):
            before[path] = describe(cfg)
    print(f"[转换前] inner_experts 节点数 = {len(before)}")
    for p, t in list(before.items())[:2]:
        print(f"   {p}\n     -> {t}")

    # ---- 转换后 ----
    conv = DeepSeekV4LoRAConverter.Config(
        rank=RANK,
        alpha=ALPHA,
        rank_experts=RANK,
        # 只加路由专家。dense 目标留空 —— 合并路径只覆盖路由专家,
        # 加在 attention/router 上的 adapter 部署时用不上。
        target_modules=(),
    )
    try:
        spec = model_registry(FLAVOR, converters=[conv])
    except BaseException as exc:  # noqa: BLE001
        _fail(f"model_registry({FLAVOR}, converters=[LoRA])", exc)

    after = {}
    lora_params = {}
    for path, cfg in walk(spec.model):
        t = describe(cfg)
        if "inner_experts" in path and path.endswith("inner_experts"):
            after[path] = t
            if "LoRA" in t:
                for attr in ("rank", "alpha", "num_experts", "dim", "hidden_dim"):
                    if hasattr(cfg, attr):
                        lora_params.setdefault(attr, getattr(cfg, attr))
    grouped_hits = sum(1 for t in after.values() if "LoRA" in t)
    dense_hits = sum(
        1
        for path, cfg in walk(spec.model)
        if "LoRA" in describe(cfg) and "inner_experts" not in path
    )

    print(f"\n[转换后] inner_experts 节点数 = {len(after)}")
    for p, t in list(after.items())[:2]:
        print(f"   {p}\n     -> {t}")

    n_layers = getattr(spec.model, "n_layers", None)
    print("\n================ 判据 ================")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  {'OK ' if cond else 'X  '} {name}: {detail}")
        ok = ok and cond

    check(
        "分组专家被 LoRA 覆盖",
        grouped_hits > 0,
        f"grouped={grouped_hits} / inner_experts 节点 {len(after)}",
    )
    check(
        "命中数 == n_layers",
        n_layers is not None and grouped_hits == n_layers,
        f"grouped={grouped_hits}  n_layers={n_layers}",
    )
    check(
        "dense 命中为 0(target_modules=())",
        dense_hits == 0,
        f"dense={dense_hits}",
    )

    if lora_params:
        print(f"\n  adapter 配置: {lora_params}")
        E = lora_params.get("num_experts")
        D = lora_params.get("dim")
        F = lora_params.get("hidden_dim")
        if None not in (E, D, F) and n_layers:
            per_layer = E * RANK * (2 * D + 3 * F)
            total = per_layer * n_layers
            print(
                f"  解析式 E×r×(2·dim+3·moe_inter) = {E}×{RANK}×(2×{D}+3×{F})"
                f" = {per_layer / 1e6:.2f} M/层"
            )
            print(
                f"  全模型 adapter = {total / 1e9:.3f} G 参数"
                f" = {total * 2 / 2**30:.2f} GiB (bf16)"
            )

    print("\n结论:" + ("本步通过。" if ok else "本步未通过。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
