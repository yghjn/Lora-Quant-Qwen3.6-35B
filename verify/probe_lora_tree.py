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

import dataclasses
import logging
import os
import re
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
    else:
        # 没给 TT_ROOT 且两个包都已经装进当前环境:就用装好的,**不要再猜目录**。
        # 以前这里不管装没装都往 sys.path 最前面塞猜到的目录,于是开发机上
        # 同事那份(带未提交改动的)clone 会盖过环境里装好的干净 clone ——
        # probe 验的就不是你实际会用来训练的那份代码了。
        import importlib.util

        try:
            installed = all(
                importlib.util.find_spec(m) is not None for m in ("torchtitan", "torchtitan_npu")
            )
        except (ImportError, ValueError):
            installed = False
        if installed:
            spec = importlib.util.find_spec("torchtitan_npu")
            origin = getattr(spec, "origin", None) or ""
            return os.path.dirname(os.path.dirname(origin)) or "<已安装的包>"
        cands += [
            os.path.expanduser("~/src/torchtitan-npu"),          # 2_pip_train.sh 的默认 TT_DIR
            "/home/ma-user/work/l84414662/torchtitan-npu",       # 开发机上同事的 clone(只读,带未提交改动)
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

# upstream torchtitan 的 Qwen 模型(models/qwen3_5/model.py)在**模块顶层** import
# 这几个包,但 torchtitan 和 torchtitan-npu 的 requirements 里都没列它们 ——
# 只在 models/qwen3_5/README.md 里有一句 `pip install ... flash-linear-attention`。
_TRITON_WHL = (
    "https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/"
    "triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
)
_KNOWN_MISSING = {
    "fla": [
        "`fla` 是 flash-linear-attention 的包名。Qwen 的 gated-delta / 因果卷积算子在",
        "它的子包 fla-core 里,只装 fla-core 就够(不需要带 transformers 的整包):",
        "    pip install fla-core==0.5.2 einops",
        "钉 0.5.2:它的 [npu] 配套是 triton-ascend==3.2.1,和 torchtitan-npu 要的一致。",
        "fla 的算子在顶层 import triton,所以 triton-ascend 也得先装上(见下一条)。",
    ],
    "triton": [
        "`triton` 这个模块名由 triton-ascend 提供(不是 PyPI 上那个 triton)。",
        "    pip install --no-deps -i https://triton-ascend.osinfra.cn/pypi/simple triton-ascend==3.2.1",
        "那个索引连不上就直接下 wheel(cp312 / aarch64,约 188 MB)再本地装:",
        f"    {_TRITON_WHL}",
    ],
    "einops": ["fla-core 的依赖:pip install einops"],
    "attn_gym": [
        "attn-gym 版本和 torchtitan-npu 的 commit 不是一套。",
        "核对过的组合:torchtitan-npu 3f9f99a + attn-gym==0.0.9。",
    ],
}


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
        missing = (
            (getattr(exc, "name", "") or "").split(".")[0]
            if isinstance(exc, ModuleNotFoundError)
            else ""
        )
        if missing in _KNOWN_MISSING:
            out += ["", f"缺的是 `{missing}`:"] + ["   " + s for s in _KNOWN_MISSING[missing]]
        elif missing:
            out += ["", f"缺的是 `{missing}`,不在已知清单里。先试 `pip install {missing}`,"
                        "并把这个名字记进 env/2_pip_train.sh。"]
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
    """import 期的失败:环境问题,退出码 2。"""
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


def _fail_real(stage: str, exc: BaseException, compat_line: str) -> None:
    """import 全过之后、构造/转换配置树时的失败:**不是环境问题**,退出码 1。

    以前这里也走 `_fail`,于是一个真正的结论(converter 在 Qwen 的配置树上
    跑不下来)被报成"上面这些都是环境问题",退出码还是 2。
    """
    print(f"\n!! 失败于 [{stage}]: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    print("\n" + "-" * 60)
    print("   这一步 import 已经全部成功,所以**不是环境问题**。")
    print("   是 converter 在 Qwen 的配置树上没走完 —— 代码层面的结论,重装环境没用。")
    print(f"   当前:{compat_line}")
    msg = str(exc)
    if "_LoRAModelConfig" in msg and "missing" in msg:
        print()
        print("   这条具体是:convert() 的最后一行把根配置派生成写死的 DSv4 根配置类,")
        print("   Qwen 的根配置没有那几个 DSv4 专有字段。前面的树遍历和 strict 校验都已")
        print("   通过(否则走不到这一行)。本仓库 src/qwen36_lora_compat.py 就是为它写的")
        print("   —— 如果看到这条,说明 probe 没走兼容层,请确认用的是最新的本仓库。")
    print("-" * 60)
    print("\n判定:本步**未通过**(退出码 1,结论性失败)。")
    sys.exit(1)


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
    from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter  # noqa: F401
except BaseException as exc:  # noqa: BLE001
    _fail("import DeepSeekV4LoRAConverter", exc)

# 本仓库的 src/:converter 走 qwen36_lora_compat(见该文件开头的说明)。
# probe 必须和 qwen36_lora_recipe.py 用**同一个**入口,否则闸门过了、训练照样炸。
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    import qwen36_lora_compat as _compat
except BaseException as exc:  # noqa: BLE001
    _fail("import qwen36_lora_compat(本仓库 src/)", exc)

import torchtitan as _tt  # noqa: E402  —— 只为打出解析到哪个 torchtitan
from torchtitan.protocols.module import Module as _Module  # noqa: E402
from torchtitan.tools.logging import logger as _tt_logger  # noqa: E402
from torchtitan_npu.models.deepseek_v4.lora import LoRAOptions  # noqa: E402

print(f"torchtitan:     {_tt.__file__}")
print(f"converter:      {_compat.describe()}\n")


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
    # torchtitan 的 Config 全是 @dataclass(slots=True),**没有 __dict__**。
    # 以前这里只走 __dict__,结果整棵树一个子节点都遍历不到 ——
    # "[转换前] inner_experts 节点数 = 0" 就是这么来的,转换后也会是 0,
    # 判据会假阴性。dataclass 走 fields(),其余对象才退回 __dict__。
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        items = [(f.name, getattr(cfg, f.name, None)) for f in dataclasses.fields(cfg)]
    else:
        items = list(getattr(cfg, "__dict__", {}).items())
    for name, value in items:
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
        _fail_real(f"model_registry({FLAVOR}) 未加 converter", exc, _compat.describe())

    before = {}
    for path, cfg in walk(base.model):
        if "inner_experts" in path and path.endswith("inner_experts"):
            before[path] = describe(cfg)
    print(f"[转换前] inner_experts 节点数 = {len(before)}")
    for p, t in list(before.items())[:2]:
        print(f"   {p}\n     -> {t}")

    # ---- 转换后 ----
    # converter 自己会打一行 "DeepSeek-V4 LoRA: dense=%d grouped=%d rank=%d ..."。
    # 这是它**实际**替换了多少个节点的计数,比事后数树更权威,所以截下来当判据。
    captured: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap(level=logging.INFO)
    prev_level = _tt_logger.level
    _tt_logger.addHandler(cap)
    if _tt_logger.getEffectiveLevel() > logging.INFO:
        _tt_logger.setLevel(logging.INFO)
    try:
        conv = _compat.lora_converter_config(
            rank=RANK,
            alpha=ALPHA,
            rank_experts=RANK,
            # 只加路由专家。dense 目标留空 —— 合并路径只覆盖路由专家,
            # 加在 attention/router 上的 adapter 部署时用不上。
            target_modules=(),
        )
        spec = model_registry(FLAVOR, converters=[conv])
    except BaseException as exc:  # noqa: BLE001
        _fail_real(f"model_registry({FLAVOR}, converters=[LoRA])", exc, _compat.describe())
    finally:
        _tt_logger.removeHandler(cap)
        _tt_logger.setLevel(prev_level)
    root = spec.model
    print(f"[转换后] 根配置类 = {describe(root)}")

    # "这是不是一个 LoRA 配置"**不能**按类名里有没有 "LoRA" 判:工厂生成的配置类叫
    # `_make_lora_adapter_config_cls.<locals>.Config`(小写 lora),而根配置
    # `LoRAQwen35Model.Config` 和 `root.lora` 这个 converter 配置的名字里反而都带
    # "LoRA"。以前就是这么判的,结果 grouped=0、dense=2,三条判据全是假阴性。
    # 权威判据是配置类是否继承 `LoRAOptions`(rank / alpha / chunk_rows 那个 mixin)。
    def is_lora_cfg(cfg) -> bool:
        return isinstance(cfg, LoRAOptions)

    after = {}
    for path, cfg in walk(root):
        if path.endswith("inner_experts"):
            after[path] = cfg
    grouped_cfgs = [c for c in after.values() if is_lora_cfg(c)]
    grouped_hits = len(grouped_cfgs)
    all_lora = [fqn for fqn, cfg, _p, _a in root.traverse(_Module.Config, recurse=True) if is_lora_cfg(cfg)]
    dense_fqns = [fqn for fqn in all_lora if not fqn.endswith("inner_experts")]
    dense_hits = len(dense_fqns)

    print(f"[转换后] inner_experts 节点数 = {len(after)}")
    for path, c in list(after.items())[:2]:
        owner = getattr(type(c), "_owner", None)
        print(f"   {path}\n     -> {describe(c)}\n        构造的模块类: {getattr(owner, '__name__', owner)}")

    layers = getattr(root, "layers", None)
    n_layers = len(layers) if isinstance(layers, (list, tuple)) else getattr(root, "n_layers", None)
    lora_line = next((msg for msg in captured if "LoRA:" in msg and "grouped=" in msg), None)
    hit = re.search(r"dense=(\d+)\s+grouped=(\d+)\s+rank=(\d+)\s+expert_rank=(\d+)", lora_line or "")
    print(f"\nconverter 自己的计数: {lora_line or '<没截到那行日志>'}")

    print("\n================ 判据 ================")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  {'OK ' if cond else 'X  '} {name}: {detail}")
        ok = ok and bool(cond)

    check(
        "分组专家被 LoRA 覆盖",
        grouped_hits > 0,
        f"grouped={grouped_hits} / inner_experts 节点 {len(after)}",
    )
    check(
        "命中数 == 层数",
        n_layers is not None and grouped_hits == n_layers == len(after),
        f"grouped={grouped_hits}  层数={n_layers}",
    )
    check(
        "dense 命中为 0(target_modules=())",
        dense_hits == 0,
        f"dense={dense_hits}" + (f"  例: {dense_fqns[:3]}" if dense_hits else ""),
    )
    check(
        "converter 自己报的计数一致",
        hit is not None
        and int(hit.group(1)) == 0
        and int(hit.group(2)) == grouped_hits
        and int(hit.group(4)) == RANK,
        (
            f"dense={hit.group(1)} grouped={hit.group(2)} expert_rank={hit.group(4)}"
            if hit
            else "没截到 'dense=.. grouped=..' 那行"
        ),
    )
    base_owner = getattr(type(base.model), "_owner", None)
    root_owner = getattr(type(root), "_owner", None)
    check(
        "根配置带 lora 字段,且仍构造原模型类",
        hasattr(root, "lora") and isinstance(root, type(base.model)) and root_owner is base_owner and root_owner is not None,
        f"lora 字段={hasattr(root, 'lora')}  构造的模型类={getattr(root_owner, '__name__', None)}",
    )

    # ---- 真的实例化一层,核对 adapter 的参数名和参数量 ----
    # 静态看配置说明不了模块能不能建出来。build() 里是 torch.empty,不会真占满内存。
    if grouped_cfgs:
        c0 = grouped_cfgs[0]
        E, D, F, r = c0.num_experts, c0.dim, c0.hidden_dim, c0.rank
        analytic = E * r * (2 * D + 3 * F)
        try:
            mod = c0.build()
            shapes = {n: tuple(p.shape) for n, p in mod.named_parameters()}
            lora_n = sum(p.numel() for n, p in mod.named_parameters() if "lora_" in n)
            want = {
                "w13_lora_a": (E, r, D),
                "w1_lora_b": (E, F, r),
                "w3_lora_b": (E, F, r),
                "w2_lora_a": (E, r, F),
                "w2_lora_b": (E, D, r),
            }
            got = {k: shapes.get(k) for k in want}
            check(
                "实例化一层:adapter 参数名和形状",
                got == want,
                f"{len(want)} 个 adapter 张量" if got == want else f"期望 {want}  实际 {got}",
            )
            check(
                "实例化一层:adapter 参数量 == E×r×(2·dim+3·moe_inter)",
                lora_n == analytic,
                f"{lora_n} vs 解析值 {E}×{r}×(2×{D}+3×{F}) = {analytic}",
            )
            check(
                "实例化一层:基座权重仍在",
                all(k in shapes for k in ("w1_EFD", "w2_EDF", "w3_EFD")),
                f"w1_EFD={shapes.get('w1_EFD')}  w2_EDF={shapes.get('w2_EDF')}  w3_EFD={shapes.get('w3_EFD')}",
            )
            scaling = getattr(mod, "_lora_scaling", None)
            check(
                "scaling == alpha / rank",
                scaling is not None and abs(scaling - ALPHA / RANK) < 1e-9,
                f"{scaling} vs {ALPHA}/{RANK} = {ALPHA / RANK}",
            )
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            check("实例化一层", False, f"{type(exc).__name__}: {exc}")
        if n_layers:
            total = analytic * n_layers
            print(
                f"\n  每层 adapter = {analytic / 1e6:.2f} M 参数;"
                f"全模型 = {total / 1e9:.3f} G 参数 = {total * 2 / 2**30:.2f} GiB (bf16)"
            )

    print("\n结论:" + ("本步通过。" if ok else "本步未通过(退出码 1:判据不成立,不是环境问题)。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
