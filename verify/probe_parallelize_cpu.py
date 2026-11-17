#!/usr/bin/env python3
"""在 CPU 上用 8 个进程把训练配置真的并行化一遍(到加载权重之前为止)。

    python3 verify/probe_parallelize_cpu.py            # 自己拉起 8 个进程
    python3 verify/probe_parallelize_cpu.py --nproc 8 --config qwen36_35b_a3b_lora_smoke

不占 NPU、不加载权重:模型建在 meta device 上(只有形状,没有数据),进程间用 gloo。
35B 的完整配置也只要一两分钟、每个进程几百 MB。

为什么要它
----------
闸门 4 / 5 只走到"配置树能转换、单层模块能实例化、配置能构造"。2026-11-16 第一次
上卡,`Trainer.__init__` 走到 `model_spec.parallelize_fn(...)` 才报::

    ValueError: LoRA_ClampGroupedExperts.w13_lora_a has no placement declared
    in sharding_config.state_shardings.

并行化要每个参数都有切分声明,而 Qwen 的声明表里没有 adapter(见
`src/qwen36_lora_compat.py` 的 `ensure_adapter_sharding`)。这一步要有设备网格才走得
到,所以单进程的闸门拦不住。本脚本照 `torchtitan/trainer.py` 的 `__init__` 同样的
顺序走:

    解析参数 -> ParallelDims -> update_from_config -> apply_overrides
    -> meta device 上 build -> verify_module_protocol -> parallelize_fn

判据
----
1. 8 个进程的 `parallelize_fn` 都不抛异常;
2. 每层路由专家的 5 个 adapter 和 3 个基座权重,本进程持有的专家数 = 总数 / EP;
3. 可训参数只有 adapter(名字含 `lora_`),张量数 = 5 × 层数。

没验到的:真设备上的显存、权重加载、前向、反传 —— 那些只有上卡才有。

`--without-fix` 会先把 adapter 的切分声明撤掉再跑,用来确认这个探针拦得住上面那个
报错(预期:判据 1 失败,报的就是那一句)。

退出码:0 通过;1 判据不成立;2 环境问题(import 失败等)。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
RESULT_TAG = "PROBE_PARALLELIZE_RESULT"


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nproc", type=int, default=8, help="进程数,要等于配置需要的 rank 数(默认 8)")
    ap.add_argument("--config", default="qwen36_35b_a3b_lora_smoke")
    ap.add_argument("--without-fix", action="store_true", help="撤掉 adapter 的切分声明再跑(自检用)")
    args = ap.parse_args()
    # 子进程的参数走环境变量:torchrun 的参数解析会把脚本后面的 --nproc 当成它自己的。
    args.worker = os.environ.get("QWEN36_PROBE_WORKER") == "1"
    if args.worker:
        args.config = os.environ["QWEN36_PROBE_CONFIG"]
        args.without_fix = os.environ.get("QWEN36_PROBE_WITHOUT_FIX") == "1"
    return args


# ---------------------------------------------------------------- launcher
def launch(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(args.nproc),
        "--rdzv_backend", "c10d", "--rdzv_endpoint", "localhost:0",
        "--local-ranks-filter", "0", "--role", "rank", "--tee", "3",
        os.path.abspath(__file__),
    ]
    env = dict(os.environ)
    env["QWEN36_PROBE_WORKER"] = "1"
    env["QWEN36_PROBE_CONFIG"] = args.config
    env["QWEN36_PROBE_WITHOUT_FIX"] = "1" if args.without_fix else "0"
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    # 只验配置和并行化,路径给占位值即可(recipe 构造配置时就读它们)
    env["HF_ASSETS_PATH"] = "/nonexistent-hf-assets-dir"
    env["DATA_FILES"] = "/nonexistent-corpus.jsonl"
    env.pop("INITIAL_LOAD_PATH", None)
    print(f"拉起 {args.nproc} 个 CPU 进程(gloo),配置 {args.config}"
          + (",**不带** adapter 切分声明" if args.without_fix else "") + " ...", flush=True)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout or ""
    verdict = [line.split(RESULT_TAG, 1)[1].strip() for line in out.splitlines() if RESULT_TAG in line]
    keep = [
        line for line in out.splitlines()
        if "[probe]" in line or "Error" in line or "error:" in line or "Traceback" in line
    ]
    # 失败时把 rank0 的输出尾巴原样给出来,成功时只留探针自己的行
    if proc.returncode != 0 or not verdict or verdict[-1] != "PASS":
        tail = [line for line in out.splitlines() if line.startswith("[rank0]")][-45:]
        print("\n".join(tail or out.splitlines()[-45:]))
        print(f"\n结论:未通过(torchrun 退出码 {proc.returncode})。")
        return 1
    print("\n".join(line.replace("[rank0]:", "").strip() for line in keep if "[probe]" in line))
    print("\n结论:通过。")
    return 0


# ---------------------------------------------------------------- worker
def worker(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    def say(msg: str) -> None:
        if dist.get_rank() == 0:
            print(f"[probe] {msg}", flush=True)

    dist.init_process_group(backend="gloo")
    world = dist.get_world_size()

    from torchtitan.config import TORCH_DTYPE_MAP, ConfigManager
    from torchtitan.config.override import apply_overrides
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed import utils as dist_utils
    from torchtitan.tools import utils

    import qwen36_lora_compat as compat

    # 没有可见的 NPU 时,上游的 device_type 落到 "cuda",建网格会去调 torch.cuda。
    # 这里只验"并行化的声明对不对",网格建在 cpu 上即可。只改本进程里的这两个名字。
    import torchtitan.distributed.parallel_dims as _pd_mod

    if utils.device_type != "cpu":
        _pd_mod.device_type = "cpu"
        utils.device_type = "cpu"

    config = ConfigManager().parse_args(
        ["--module", "qwen36_lora_recipe", "--config", args.config, "--training.steps", "5"]
    )
    if args.without_fix:
        from torchtitan.models.qwen3_5 import sharding as qwen_sharding

        for name in compat.ADAPTER_PARAMS:
            qwen_sharding._GROUPED_EXPERTS_PARAM_LAYOUT.pop(name, None)

    spec = config.model_spec
    par = config.parallelism
    parallel_dims = ParallelDims.from_config(par, world)
    dist_utils.set_spmd_backend(par.spmd_backend)
    say(f"world={world} dp_shard={parallel_dims.dp_shard} tp={parallel_dims.tp} ep={parallel_dims.ep}"
        f" spmd_backend={par.spmd_backend} device={utils.device_type}")

    model_config = spec.model
    model_config.update_from_config(config=config)
    if config.override.imports:
        apply_overrides(config.override, config)
    with torch.device("meta"), utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
        model = model_config.build()
    model.verify_module_protocol()
    n_layers = len(model.layers)
    say(f"meta 模型已建: {n_layers} 层")

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        say(f"{'OK ' if cond else 'X  '} {name}" + (f": {detail}" if detail else ""))

    try:
        out = spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=par,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )
        model = out if out is not None else model
        check("parallelize_fn 走完", True)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        check("parallelize_fn 走完", False, f"{type(exc).__name__}: {exc}")
        print(f"{RESULT_TAG} FAIL", flush=True)
        return 1

    # 每层路由专家:本进程持有的专家数
    names = tuple(compat.ADAPTER_PARAMS) + ("w1_EFD", "w2_EDF", "w3_EFD")
    total = model_config.layers[0].moe.routed_experts.inner_experts.num_experts
    want = total // max(parallel_dims.ep, 1)
    bad: list[str] = []
    seen = 0
    # model.layers 是 ModuleDict(键是层号字符串),直接迭代拿到的是键
    layers = list(model.layers.values()) if hasattr(model.layers, "values") else list(model.layers)
    for i, layer in enumerate(layers):
        experts = layer.moe.routed_experts.inner_experts
        for name in names:
            p = getattr(experts, name, None)
            if p is None:
                bad.append(f"layer{i}.{name} 不存在")
                continue
            seen += 1
            local = p.to_local() if hasattr(p, "to_local") else p
            if local.shape[0] != want:
                bad.append(f"layer{i}.{name} 本地专家数 {local.shape[0]} != {want}")
    check(f"每层 8 个专家参数,本进程各持有 {want}/{total} 个专家", not bad and seen == 8 * n_layers,
          f"{seen} 个张量" + (f";问题: {bad[:3]}" if bad else ""))

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_adapter = [n for n in trainable if "lora_" not in n]
    check("可训参数只有 adapter", not non_adapter and len(trainable) == 5 * n_layers,
          f"{len(trainable)} 个可训张量(期望 {5 * n_layers})" + (f";混入: {non_adapter[:3]}" if non_adapter else ""))

    # 保存器(qwen36_adapter_save)会按这里的名字和 placements 落盘,merge_lora 照着读。
    # 2026-11-17 第一批真分片的名字带 `_checkpoint_wrapped_module`,merge_lora 认不出来 ——
    # 这一条用并行化之后的**真 DTensor** 把两头对一遍,不用等到训完才发现。
    import importlib.util

    from qwen36_adapter_save import canonical_name

    spec_m = importlib.util.spec_from_file_location("_merge_lora", os.path.join(_SRC, "merge_lora.py"))
    merge = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(merge)
    adapters = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    unknown = [n for n, _ in adapters if not merge._ADAPTER_RE.search(canonical_name(n))]
    placements = sorted({str(p.placements).replace(" ", "") for _, p in adapters if hasattr(p, "placements")})

    def _concat_ok(text: str) -> bool:
        strided = merge._STRIDED_RE.search(text)
        return "Shard(dim=0)" in text and (strided is None or int(strided.group(1)) == world)

    check("adapter 落盘用的名字和切分方式,merge_lora 都认得",
          bool(adapters) and not unknown and bool(placements) and all(_concat_ok(t) for t in placements),
          f"原名例 {adapters[0][0] if adapters else '-'};placements {placements}"
          + (f";认不出 {unknown[:2]}" if unknown else ""))

    print(f"{RESULT_TAG} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    args = _parse()
    if not args.worker:
        return launch(args)
    try:
        return worker(args)
    except ModuleNotFoundError as exc:
        traceback.print_exc()
        print(f"[probe] 缺模块 {exc.name} —— 环境问题,先让 env/2_pip_train.sh 的五道闸门过了。", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
