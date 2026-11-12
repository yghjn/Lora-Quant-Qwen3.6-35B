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
    TT_ROOT=<torchtitan-npu 目录> python3 verify/probe_lora_tree.py

装好了的环境(`env/2_pip_train.sh` 跑过)不需要 `TT_ROOT`,直接跑即可。
本脚本会自己把 `<TT_ROOT>` 和 `<TT_ROOT>/third_party/torchtitan` 接上
`sys.path`,不需要你手动配 PYTHONPATH。

环境变量:
    TT_ROOT     torchtitan-npu 的目录。不给就按常见位置猜,猜不到且 import
                失败时会把原因和两种修法打出来
    TT_FLAVOR   默认 35B-A3B
    TT_RANK     默认 16
    TT_ALPHA    默认 32

两个环境前提(踩过,所以脚本自己查)
------------------------------------
* **Python ≥ 3.12。** `torchtitan-npu/pyproject.toml` 写的是
  `requires-python = ">=3.12"`(upstream torchtitan 是 >=3.11)。
  在 3.9 上 import 会报 `dataclass() got an unexpected keyword argument
  'kw_only'` —— 那是 Python 版本,不是 torchtitan 的 bug,`kw_only` 是 3.10 才有的。
* **upstream torchtitan 要单独装。** `third_party/` 被 torchtitan-npu 自己的
  `.gitignore` 挡掉(第 115 行 `/third_party/`),而且它**不是 submodule**
  —— `.gitmodules` 里只有 `third_party/agent-skills/KPBot` 一条。
  所以 clone 下来 `third_party/` 是空的,`pip install -e <torchtitan-npu>`
  也只装 `torchtitan_npu`(`packages.find` 里 `exclude = ["third_party*"]`)。
  `import torchtitan` 报 No module named 就是这个原因。
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


# torchtitan-npu 的两个来源。装的时候要分别装,见模块 docstring。
TT_NPU_GIT = "https://gitcode.com/cann/torchtitan-npu.git"
TT_GIT = "https://gitcode.com/GitHub_Trending/to/torchtitan.git"
TT_PIN = "086bf6c16"  # torchtitan 0.3.0,本项目核对过的那个 commit


def _wire_paths() -> str | None:
    """把 torchtitan-npu 的两个包目录接上 sys.path。返回用的是哪个 root。

    装好了就什么都不做(import 走已安装的包)。给 TT_ROOT 时用它;没给就猜。
    **不碰 PYTHONPATH** —— 只改本进程的 sys.path,免得影响后面的子进程。
    """
    cands = []
    if os.environ.get("TT_ROOT"):
        cands.append(os.environ["TT_ROOT"])
    cands += [
        os.path.expanduser("~/src/torchtitan-npu"),          # 2_pip_train.sh 的默认 TT_DIR
        "/home/ma-user/work/l84414662/torchtitan-npu",       # 本机同事的 clone(只读)
    ]
    for root in cands:
        up = os.path.join(root, "third_party", "torchtitan")
        if os.path.isdir(os.path.join(root, "torchtitan_npu")) and os.path.isdir(
            os.path.join(up, "torchtitan")
        ):
            for p in (root, up):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return root
        # root 对但 third_party 空 —— 这是最常见的那种坏法,单独说
        if os.path.isdir(os.path.join(root, "torchtitan_npu")) and not os.path.isdir(up):
            print(f"注意:{root} 在,但 {up} 不存在(third_party 被 gitignore 了)。")
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


TT_ROOT_USED = _wire_paths()


def _env_hints(exc: BaseException | None = None) -> list[str]:
    """给 import 失败时附的环境诊断。按"最可能是这个"排序。

    `exc` 用来挑该说哪一段:`kw_only` 那种是 Python 版本,和 third_party 无关,
    把"No module named 的根因"整段也打出来只是噪音。
    """
    missing_tt = isinstance(exc, ModuleNotFoundError) and (
        getattr(exc, "name", "") or ""
    ).split(".")[0] in ("torchtitan", "torchtitan_npu")
    out = []
    if sys.version_info < (3, 12):
        out += [
            f"Python 是 {sys.version_info.major}.{sys.version_info.minor},"
            "torchtitan-npu 要求 >= 3.12(upstream torchtitan >= 3.11)。",
            "   报 `dataclass() got an unexpected keyword argument 'kw_only'` 就是这个 ——",
            "   kw_only 是 Python 3.10 才有的,和 torchtitan 无关。",
            "",
        ]
    if TT_ROOT_USED is None:
        out += [
            "没找到 torchtitan-npu 的目录,也没在已安装的包里找到 torchtitan。",
            f"   给路径:TT_ROOT=<torchtitan-npu 目录> python3 {os.path.basename(__file__)}",
            "",
        ]
    if not missing_tt and TT_ROOT_USED is not None:
        # 路径是通的、错的是别的东西(版本之类)。下面那段就不用说了。
        out += ["(路径没问题:torchtitan 解析到了。上面那条才是真因。)"]
        return out
    out += [
        "`import torchtitan` 报 No module named 的根因(不是漏装一个包那么简单):",
        "   upstream torchtitan 在 <root>/third_party/torchtitan,而 third_party 被",
        "   torchtitan-npu 自己的 .gitignore 挡掉,**且不是 submodule**",
        "   (.gitmodules 里只有 agent-skills/KPBot)。所以:",
        "     - clone torchtitan-npu 下来,third_party/ 是空的;",
        "     - `git submodule update --init` 拿不到它;",
        "     - `pip install -e <root>` 只装 torchtitan_npu",
        "       (pyproject 的 packages.find 里 exclude = [\"third_party*\"])。",
        "",
        "   两种修法 —— 装:",
        f"     git clone {TT_GIT} <root>/third_party/torchtitan",
        f"     git -C <root>/third_party/torchtitan checkout {TT_PIN}",
        "     pip install --no-deps -e <root>/third_party/torchtitan",
        "     pip install --no-deps -e <root>",
        "   (--no-deps 是必须的:torchtitan 钉 torch==2.14.0.dev20260719,",
        "    让 pip 去解会把环境拆了。)",
        "",
        "   或者不装,只给路径:",
        f"     TT_ROOT=<root> python3 {os.path.basename(__file__)}",
    ]
    return out


def _fail(stage: str, exc: BaseException) -> None:
    print(f"\n!! 失败于 [{stage}]: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    print("\n" + "-" * 60)
    for line in _env_hints(exc):
        print("   " + line if line else "")
    print("-" * 60)
    print(
        "\n判定:本步**未通过**,但注意 import 失败不等于结论为假 —— "
        "上面这些都是环境问题,\n"
        "      要在装对的 venv 里重跑再下 \"LoRA 挂不上 Qwen 配置树\" 这个结论。"
    )
    sys.exit(2)


# ---------------------------------------------------------------- imports
if TT_ROOT_USED:
    print(f"torchtitan root: {TT_ROOT_USED}")
print(f"Python: {sys.version.split()[0]}"
      + ("" if sys.version_info >= (3, 12) else "   !! torchtitan-npu 要求 >= 3.12"))

try:
    from torchtitan.models.qwen3_5 import model_registry, qwen3_5_configs
except BaseException as exc:  # noqa: BLE001
    _fail("import torchtitan.models.qwen3_5", exc)

try:
    from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter
except BaseException as exc:  # noqa: BLE001
    _fail("import DeepSeekV4LoRAConverter", exc)

import torchtitan as _tt  # noqa: E402  —— 只为打出解析到哪个 torchtitan

print(f"torchtitan:     {_tt.__file__}\n")


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
