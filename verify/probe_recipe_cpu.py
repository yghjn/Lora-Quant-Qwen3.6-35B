#!/usr/bin/env python3
"""验证本仓库的训练配置(src/qwen36_lora_recipe.py)在当前环境里能 import、能构造。

    python3 verify/probe_recipe_cpu.py

纯 CPU,不加载权重,不占卡。`verify/probe_lora_tree.py`(工单 §0.1)只验 converter;
这个脚本验的是**训练真正会走的那条 import 链** —— `MODULE=qwen36_lora_recipe`
启动时做的事,到"建模型"之前为止。

为什么单独要它:闸门 0.1 的入口是 `torchtitan.models.qwen3_5`,而训练的入口是
`torchtitan_npu.models.qwen3_5`,后者多 import 一截多模态数据管线,需要
`torchvision`(以及 PIL)。0.1 过了不代表训练起得来 —— 2026-11-16 在一个全新
环境里实跑才发现这一点,而且 adapter 落盘钩子依赖的文件在干净 clone 里不存在。

判据
----
1. recipe 能 import(缺包会在这里报,并说明缺什么);
2. 三个配置函数都能构造出 `Trainer.Config`;
3. 每个配置的 40 层路由专家都是 LoRA 配置,rank / alpha 与配置名一致;
4. adapter 落盘钩子能解析到本仓库的 `qwen36_adapter_save`
   (没设 `ADAPTER_SAVE_DIR` 时抛 RuntimeError,而不是 ModuleNotFoundError)。

退出码:0 通过;1 判据不成立;2 import 期失败(环境问题)。
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")
# 构造配置不读这两个,但 recipe / 上游的配置函数可能在缺省时告警,给个占位值
os.environ.setdefault("HF_ASSETS_PATH", "/nonexistent-tokenizer-dir")
os.environ.setdefault("DATA_FILES", "/nonexistent-corpus.jsonl")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_KNOWN = {
    "torchvision": [
        "`torchtitan_npu.models.qwen3_5` 会 import 多模态数据管线,需要 torchvision。",
        "要和 torch 的 nightly 成对(torchtitan-npu 的 requirements.txt 钉的是 0.29.0.dev20260720):",
        "  pip install --no-deps 'https://download.pytorch.org/whl/nightly/cpu/"
        "torchvision-0.29.0.dev20260720%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl'",
        "  pip install pillow",
        "**一定带 --no-deps**:torchvision 声明依赖 torch,让 pip 去解会换掉装好的 nightly。",
    ],
    "PIL": ["pip install pillow"],
    "fla": ["pip install fla-core==0.5.2 einops   (见 verify/probe_lora_tree.py 的说明)"],
    "triton": ["由 triton-ascend 提供:见 env/2_pip_train.sh 里的 TRITON_WHL"],
    "einops": ["pip install einops"],
    "requests": ["pip install requests"],
}


def _import_fail(stage: str, exc: BaseException) -> None:
    print(f"\n!! 失败于 [{stage}]: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    missing = (getattr(exc, "name", "") or "").split(".")[0] if isinstance(exc, ModuleNotFoundError) else ""
    print("\n" + "-" * 60)
    if missing in _KNOWN:
        print(f"   缺的是 `{missing}`:")
        for line in _KNOWN[missing]:
            print("      " + line)
    elif missing:
        print(f"   缺的是 `{missing}`,不在已知清单里。先试 pip install {missing},并记进 env/2_pip_train.sh。")
    else:
        print("   不是缺包。看上面的 traceback。")
    print("-" * 60)
    print("\n判定:未通过(退出码 2,import 期失败 = 环境问题)。")
    sys.exit(2)


try:
    import qwen36_lora_recipe as R
except BaseException as exc:  # noqa: BLE001
    _import_fail("import qwen36_lora_recipe", exc)

from torchtitan_npu.models.deepseek_v4.lora import LoRAOptions  # noqa: E402

import qwen36_lora_compat as _compat  # noqa: E402

print(f"converter: {_compat.describe()}")

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  {'OK ' if cond else 'X  '} {name}" + (f": {detail}" if detail else ""))
    ok = ok and bool(cond)


EXPECT = {
    "qwen36_35b_a3b_lora_smoke": (16, 32.0),
    "qwen36_35b_a3b_lora_r16": (16, 32.0),
    "qwen36_35b_a3b_lora_r4": (4, 8.0),
}
last = None
for fn, (rank, alpha) in EXPECT.items():
    print(f"\n== {fn}")
    try:
        cfg = getattr(R, fn)()
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        check("能构造 Trainer.Config", False, f"{type(exc).__name__}: {exc}")
        continue
    last = cfg
    spec = cfg.model_spec
    root = spec.model
    layers = list(getattr(root, "layers", []))
    lora_layers = [l for l in layers if isinstance(l.moe.routed_experts.inner_experts, LoRAOptions)]
    check("能构造 Trainer.Config", True, f"steps={cfg.training.steps}")
    check("每层路由专家都是 LoRA 配置", len(layers) > 0 and len(lora_layers) == len(layers),
          f"{len(lora_layers)} / {len(layers)}")
    if lora_layers:
        ie = lora_layers[0].moe.routed_experts.inner_experts
        check("rank / alpha 与配置名一致", (ie.rank, float(ie.alpha)) == (rank, alpha),
              f"rank={ie.rank} alpha={ie.alpha}(期望 {rank} / {alpha})")
    check("parallelize_fn 是带冻结的那个", getattr(spec.parallelize_fn, "__name__", "") == "_parallelize_lora",
          str(getattr(spec.parallelize_fn, "__name__", spec.parallelize_fn)))
    check("挂了 adapter 落盘钩子", getattr(spec, "post_optimizer_build_fn", None) is not None)

print("\n== adapter 落盘钩子的 import 路径")
if last is not None:
    saved = os.environ.pop("ADAPTER_SAVE_DIR", None)
    try:
        last.model_spec.post_optimizer_build_fn([], [], None)
        check("没设 ADAPTER_SAVE_DIR 时抛 RuntimeError", False, "没抛异常")
    except RuntimeError as exc:
        check("没设 ADAPTER_SAVE_DIR 时抛 RuntimeError(saver 解析到了)", "ADAPTER_SAVE_DIR" in str(exc), str(exc)[:50])
    except BaseException as exc:  # noqa: BLE001
        check("没设 ADAPTER_SAVE_DIR 时抛 RuntimeError", False, f"{type(exc).__name__}: {exc}")
    finally:
        if saved is not None:
            os.environ["ADAPTER_SAVE_DIR"] = saved
else:
    check("至少有一个配置构造成功", False)

print("\n结论:" + ("通过。" if ok else "未通过(退出码 1)。"))
sys.exit(0 if ok else 1)
