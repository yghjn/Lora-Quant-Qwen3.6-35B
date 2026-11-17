#!/usr/bin/env python3
"""src/qwen36_adapter_save.py 的 CPU 自测。只需要 torch + safetensors,不需要 torchtitan。

    python3 verify/test_adapter_save_cpu.py

验的是落盘契约 —— `src/merge_lora.py` 是照这个契约读的:
文件名、只收 lora_* 参数、bf16、metadata 里的 step / rank。
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

# 纯 CPU 自测。环境里装着 torch_npu 时,`import torch` 会自动加载它,而它给
# 优化器打的补丁会在第一次 step() 时去初始化 NPU —— 没给卡就直接炸在 aclInit。
# 这和被测的东西无关,关掉自动加载。必须在 import torch 之前设。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import qwen36_adapter_save as S  # noqa: E402

# merge_lora.py 里认分片用的正则,抄过来对拍;改了那边这里会跟着红
SHARD_RE = re.compile(r"adapter_r(\d+)_step(\d+)\.safetensors$")

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1_EFD = torch.nn.Parameter(torch.randn(4, 6, 8))
        self.w13_lora_a = torch.nn.Parameter(torch.randn(4, 2, 8))
        self.w1_lora_b = torch.nn.Parameter(torch.zeros(4, 6, 2))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_experts = Experts()
        self.dense = torch.nn.Linear(8, 8)


def run(every: int, steps: int, out_dir: str | None, n_opt: int = 1):
    net = Net()
    opts = [torch.optim.AdamW([p for n, p in net.named_parameters() if "lora_" in n], lr=1e-2) for _ in range(n_opt)]
    os.environ.pop("ADAPTER_SAVE_DIR", None)
    if out_dir is not None:
        os.environ["ADAPTER_SAVE_DIR"] = out_dir
    os.environ["ADAPTER_SAVE_EVERY"] = str(every)
    ok = S.register_adapter_saver(opts, [net])
    for _ in range(steps):
        for o in opts:
            o.zero_grad()
        loss = (net.inner_experts.w13_lora_a.sum() + net.inner_experts.w1_lora_b.sum())
        loss.backward()
        for o in opts:
            o.step()
    return ok, net


print("== A. 没设 ADAPTER_SAVE_DIR")
ok, _ = run(every=2, steps=3, out_dir=None)
check("返回 False(recipe 会把它升级成异常)", ok is False)

print("== B. 每 2 步存一次,跑 5 步")
with tempfile.TemporaryDirectory() as d:
    ok, net = run(every=2, steps=5, out_dir=d)
    files = sorted(os.listdir(d))
    check("返回 True", ok is True)
    check("恰好存了 step 2 和 step 4", files == [S.shard_name(0, 2), S.shard_name(0, 4)], str(files))
    check("文件名对得上 merge_lora 的正则", all(SHARD_RE.search(f) for f in files), str(files))
    m = SHARD_RE.search(files[-1])
    check("正则解出 rank=0 step=4", m is not None and (int(m.group(1)), int(m.group(2))) == (0, 4))
    with safe_open(os.path.join(d, files[-1]), framework="pt") as f:
        keys = sorted(f.keys())
        meta = f.metadata() or {}
        t = {k: f.get_tensor(k) for k in keys}
    check("只收 lora_* 参数", keys == ["inner_experts.w13_lora_a", "inner_experts.w1_lora_b"], str(keys))
    check("基座权重没被写进去", not any("w1_EFD" in k or "dense" in k for k in keys))
    check("全部是 bf16", all(v.dtype == torch.bfloat16 for v in t.values()), str({k: v.dtype for k, v in t.items()}))
    check("形状不变", tuple(t["inner_experts.w13_lora_a"].shape) == (4, 2, 8))
    check("metadata 有 step / rank", meta.get("step") == "4" and meta.get("rank") == "0", str(meta))
    live = net.inner_experts.w13_lora_a.detach().to(torch.bfloat16)
    with safe_open(os.path.join(d, files[0]), framework="pt") as f:
        early = f.get_tensor("inner_experts.w13_lora_a")
    check("step 4 的值不同于 step 2(确实在训练中途各存了一次)", not torch.equal(early, t["inner_experts.w13_lora_a"]))
    check("普通张量不写 global_shape(只有 DTensor 才写)", not any(k.endswith(".global_shape") for k in meta))
    del live

print("== C. 两个优化器:钩子只挂一个,步数不翻倍")
with tempfile.TemporaryDirectory() as d:
    ok, _ = run(every=2, steps=4, out_dir=d, n_opt=2)
    files = sorted(os.listdir(d))
    check("仍然是 step 2 和 step 4", files == [S.shard_name(0, 2), S.shard_name(0, 4)], str(files))

print("== D. 没有 lora_ 参数的模型")
with tempfile.TemporaryDirectory() as d:
    n = S.save_adapters([torch.nn.Linear(4, 4)], d, 1)
    check("返回 0 且不写空文件", n == 0 and os.listdir(d) == [], str(os.listdir(d)))

print("== E. ADAPTER_SAVE_EVERY 非法")
with tempfile.TemporaryDirectory() as d:
    os.environ["ADAPTER_SAVE_DIR"] = d
    os.environ["ADAPTER_SAVE_EVERY"] = "0"
    try:
        S.register_adapter_saver([torch.optim.SGD(Net().parameters(), lr=0.1)], [Net()])
        check("EVERY=0 被拒绝", False, "没抛异常")
    except ValueError:
        check("EVERY=0 被拒绝", True)

print("== F. 模块被 activation checkpoint 包了一层(真训练就是这样)")
# 训练时每一层都套了 torch 的 CheckpointWrapper,named_parameters() 给出的名字中间
# 多一段 `_checkpoint_wrapped_module`。2026-11-17 第一批真分片带着它,merge_lora.py
# 把 200 个 adapter 全判成了"不在路由专家上"。落盘的键必须是规范名字。
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper  # noqa: E402


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = torch.nn.Module()
        self.moe.routed_experts = torch.nn.Module()
        self.moe.routed_experts.inner_experts = Experts()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleDict({"0": checkpoint_wrapper(Layer()), "1": checkpoint_wrapper(Layer())})


with tempfile.TemporaryDirectory() as d:
    model = Model()
    raw = [n for n, _ in model.named_parameters() if "lora_" in n]
    check("前提:named_parameters() 的名字确实带那一段",
          all("_checkpoint_wrapped_module" in n for n in raw) and len(raw) == 4, str(raw[:1]))
    S.save_adapters([model], d, 5)
    with safe_open(os.path.join(d, S.shard_name(0, 5)), framework="pt") as f:
        keys = sorted(f.keys())
    want = sorted(
        f"layers.{i}.moe.routed_experts.inner_experts.{n}" for i in (0, 1) for n in ("w13_lora_a", "w1_lora_b")
    )
    check("落盘的键是规范名字(不带包装段)", keys == want, str(keys))
    check("和 state_dict 的键一致", set(keys) <= set(model.state_dict()), str(keys[:1]))

    # merge_lora.py 认 adapter 用的正则和规范化清单,直接从那边取来对拍
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_merge_lora", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "merge_lora.py")
    )
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    check("merge_lora 的正则认得每一个落盘的键", all(M._ADAPTER_RE.search(k) for k in keys), str(keys[:1]))
    check("两边的包装段清单一致", tuple(M.WRAPPER_SEGMENTS) == tuple(S.WRAPPER_SEGMENTS))
    check("merge_lora 读旧分片时也能规范化(原名 -> 规范名)",
          sorted(M.canonical_name(n) for n in raw) == want)

print(f"\n{passed} 通过, {failed} 失败")
sys.exit(1 if failed else 0)
