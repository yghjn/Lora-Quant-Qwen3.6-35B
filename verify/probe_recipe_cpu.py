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
   (没设 `ADAPTER_SAVE_DIR` 时抛 RuntimeError,而不是 ModuleNotFoundError);
5. **基座权重会被加载**:`checkpoint.enable` / `initial_load_in_hf` 为真,
   `hf_assets_path` 等于 `HF_ASSETS_PATH`,且 `load_only` 为真(不存全模型)。
   上游这个开关默认是关的,关着的时候训练照样 exit 0,只是基座是随机的;
6. NPU 的 GDN kernel override 在 `override.imports` 里,并且那个名字真的存在;
   权重适配器是本仓库的 `Qwen36StateDictAdapter`;MoE 负载均衡偏置已关;
7. 把训练启动器的参数解析原样走一遍:`--training.steps 5` 生效,
   前面几条在解析之后仍然成立。

退出码:0 通过;1 判据不成立;2 import 期失败(环境问题)。
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")
# recipe 构造配置时就读这两个(缺了直接抛)。这里只验配置,固定用占位值 ——
# 不用 setdefault:shell 里若已 export 了真路径,判据里比对的值就不确定了。
os.environ["HF_ASSETS_PATH"] = "/nonexistent-hf-assets-dir"
os.environ["DATA_FILES"] = "/nonexistent-corpus.jsonl"
os.environ.pop("INITIAL_LOAD_PATH", None)

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
    ck = cfg.checkpoint
    check("基座权重会被加载(checkpoint.enable + initial_load_in_hf)",
          ck.enable is True and ck.initial_load_in_hf is True and ck.initial_load_model_only is True,
          f"enable={ck.enable} initial_load_in_hf={ck.initial_load_in_hf}")
    check("权重和 tokenizer 来自 HF_ASSETS_PATH",
          cfg.hf_assets_path == os.environ["HF_ASSETS_PATH"] and ck.initial_load_path is None,
          f"hf_assets_path={cfg.hf_assets_path} initial_load_path={ck.initial_load_path}")
    check("不保存全模型(load_only)", ck.load_only is True, f"load_only={ck.load_only}")
    check("NPU GDN override 在 override.imports 里", R.NPU_GDN_OVERRIDE in list(cfg.override.imports),
          str(list(cfg.override.imports)))
    check("权重适配器是 Qwen3.6 的(专家权重 [out, in] 排布)",
          getattr(spec.state_dict_adapter, "__name__", "") == "Qwen36StateDictAdapter",
          str(getattr(spec.state_dict_adapter, "__name__", spec.state_dict_adapter)))
    lb = [l.moe.load_balance_coeff for l in layers if getattr(l, "moe", None) is not None]
    check("MoE 负载均衡偏置已关(训练时的路由 = 部署时的路由)", bool(lb) and all(v is None for v in lb),
          f"{sum(v is None for v in lb)} / {len(lb)} 层为 None")

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

print("\n== 总步数(给 adapter 保存器用:最后一步一定保存)")
_argv = list(sys.argv)
try:
    sys.argv = ["train"]
    a = R._effective_steps(256)
    sys.argv = ["train", "--module", "x", "--training.steps", "5"]
    b = R._effective_steps(256)
    sys.argv = ["train", "--training.steps=7"]
    c = R._effective_steps(256)
finally:
    sys.argv = _argv
check("命令行没给 --training.steps 时用配置里的值;给了以命令行为准(两种写法)",
      (a, b, c) == (256, 5, 7), f"{(a, b, c)}(期望 (256, 5, 7))")

print("\n== Triton 的编译器包装")
_cc = os.environ.get("CC", "")
check("导入 recipe 之后 CC 指向 env/triton_cxx.sh,且可执行",
      _cc.replace("\\", "/").endswith("env/triton_cxx.sh") and os.access(_cc, os.X_OK),
      f"CC={_cc or '(没设)'}")

print("\n== 权重路径的环境变量")
_SMOKE = "qwen36_35b_a3b_lora_smoke"
_PLACEHOLDER = os.environ["HF_ASSETS_PATH"]


def _raises(value: str | None, want: str) -> tuple[bool, str]:
    if value is None:
        os.environ.pop("HF_ASSETS_PATH", None)
    else:
        os.environ["HF_ASSETS_PATH"] = value
    try:
        getattr(R, _SMOKE)()
        return False, "没抛异常"
    except RuntimeError as exc:
        return want in str(exc), str(exc)[:46]
    except BaseException as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        os.environ["HF_ASSETS_PATH"] = _PLACEHOLDER


good, msg = _raises(None, "HF_ASSETS_PATH")
check("没设 HF_ASSETS_PATH 时拒绝构造(而不是在随机基座上训)", good, msg)
good, msg = _raises("assets/hf/x", "绝对路径")
check("HF_ASSETS_PATH 是相对路径时拒绝构造", good, msg)
os.environ["INITIAL_LOAD_PATH"] = "/nonexistent-weights-dir"
try:
    c2 = getattr(R, _SMOKE)()
    check("INITIAL_LOAD_PATH 给了就从它加载权重",
          c2.checkpoint.initial_load_path == "/nonexistent-weights-dir" and c2.hf_assets_path == _PLACEHOLDER,
          f"initial_load_path={c2.checkpoint.initial_load_path}")
except BaseException as exc:  # noqa: BLE001
    check("INITIAL_LOAD_PATH 给了就从它加载权重", False, f"{type(exc).__name__}: {exc}")
finally:
    os.environ.pop("INITIAL_LOAD_PATH", None)

print("\n== NPU GDN override 这个名字存在吗")
try:
    import importlib

    _mod_name, _attr = R.NPU_GDN_OVERRIDE.rsplit(".", 1)
    _mod = importlib.import_module(_mod_name)
    check(f"{_mod_name} 能 import,且有 `{_attr}`", hasattr(_mod, _attr))
except BaseException as exc:  # noqa: BLE001
    traceback.print_exc()
    check("override 模块能 import", False, f"{type(exc).__name__}: {exc}")

print("\n== 按训练启动器的方式解析参数")
# scripts/run_train.sh 最终执行的是
#   python -m torchtitan_npu.train --module <MODULE> --config <CONFIG> "$@"
# 这里把同一段解析走一遍:命令行覆盖生效,且不会把上面几项冲掉。
try:
    from torchtitan.config import ConfigManager

    parsed = ConfigManager().parse_args(
        ["--module", "qwen36_lora_recipe", "--config", _SMOKE, "--training.steps", "5"]
    )
    check("--training.steps 5 生效", parsed.training.steps == 5, f"steps={parsed.training.steps}")
    check("解析之后权重加载设置还在",
          parsed.checkpoint.enable is True and parsed.checkpoint.initial_load_in_hf is True
          and parsed.checkpoint.load_only is True and parsed.hf_assets_path == _PLACEHOLDER,
          f"enable={parsed.checkpoint.enable} hf_assets_path={parsed.hf_assets_path}")
    check("解析之后 override 还在", R.NPU_GDN_OVERRIDE in list(parsed.override.imports),
          str(list(parsed.override.imports)))
except BaseException as exc:  # noqa: BLE001
    traceback.print_exc()
    check("启动器的参数解析能走通", False, f"{type(exc).__name__}: {exc}")

print("\n结论:" + ("通过。" if ok else "未通过(退出码 1)。"))
sys.exit(0 if ok else 1)
