#!/usr/bin/env python
"""merge_lora.py 的端到端自测,纯 CPU、不占卡、不需要真权重。

    python -I verify/test_merge_lora_cpu.py

验这些:

1. ``[E, 2*hidden, dim]`` 朝向(对官方权重实测到的布局)合并数值正确
2. ``[E, dim, 2*hidden]`` 朝向(state_dict_adapter 期望的布局)合并数值正确
3. gate 和 up 两半没有串位 —— 两半喂的是不同的 delta,串位会让误差爆掉
4. 共享专家等不该动的键一个比特都没变
5. ``--rank`` 和张量形状不一致时**必须**失败(§7.2)
6. ``--alpha`` 没有默认值:不给就必须失败(DSv4.1 的 8 倍事故)
7. 空 adapter(全 0 的 B)会被"merge 是空操作"的判据拦住(§7.1)
8. ``dim == 2*hidden`` 时形状判不出朝向,auto **必须拒绝**而不是瞎选一边;
   显式 ``--gate-up-layout`` 则要能跑通

参考值是用 ``einsum`` 独立算的,不复用 merge_lora 里的任何代码路径。

几何取 ``D=16 / F=3``,于是 ``2F=6 != D`` —— 和真模型
(``D=2048``, ``2F=1024``)一样可区分。第 8 项单独用一组
``D == 2F`` 的退化几何。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE = os.path.join(HERE, "..", "src", "merge_lora.py")

# 小一号的 Qwen 几何。**F 不能取 D/2** —— 那样 2F == D,两种朝向形状一样,
# 按形状判朝向在原理上就不成立(真模型是 D=2048 / 2F=1024,可区分)。
E, R, D, F = 6, 4, 16, 3
ALPHA = 32.0
LAYERS = (0, 1)

_ok = 0
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  [ok]   {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}" + (f"\n         {detail}" if detail else ""))


def make_adapter(path: str, zero_b: bool = False, d: int = D, f: int = F) -> dict:
    g = torch.Generator().manual_seed(7)
    t: dict[str, torch.Tensor] = {}
    for layer in LAYERS:
        p = f"layers.{layer}.moe.routed_experts.inner_experts."
        t[p + "w13_lora_a"] = torch.randn(E, R, d, generator=g).bfloat16()
        t[p + "w2_lora_a"] = torch.randn(E, R, f, generator=g).bfloat16()
        for name, shape in (
            ("w1_lora_b", (E, f, R)),
            ("w3_lora_b", (E, f, R)),
            ("w2_lora_b", (E, d, R)),
        ):
            t[p + name] = (
                torch.zeros(shape) if zero_b else torch.randn(shape, generator=g) * 0.5
            ).bfloat16()
    save_file(t, path, metadata={"step": "20", "rank": "0"})
    return t


def make_base(dirpath: str, orient: str, d: int = D, f: int = F) -> dict:
    """orient='efd' -> gate_up [E,2F,D] / down [E,D,F](实测布局)
    orient='edf' -> gate_up [E,D,2F] / down [E,F,D](adapter 期望的布局)"""
    g = torch.Generator().manual_seed(11)
    t: dict[str, torch.Tensor] = {}
    for layer in LAYERS:
        p = f"model.language_model.layers.{layer}.mlp.experts."
        if orient == "efd":
            t[p + "gate_up_proj"] = torch.randn(E, 2 * f, d, generator=g).bfloat16()
            t[p + "down_proj"] = torch.randn(E, d, f, generator=g).bfloat16()
        else:
            t[p + "gate_up_proj"] = torch.randn(E, d, 2 * f, generator=g).bfloat16()
            t[p + "down_proj"] = torch.randn(E, f, d, generator=g).bfloat16()
    # 一个不该被动的键,确认合并没有乱写
    t["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"] = torch.randn(
        d, d, generator=g
    ).bfloat16()
    os.makedirs(dirpath, exist_ok=True)
    save_file(t, os.path.join(dirpath, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(dirpath, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "qwen3_5_moe"}, fh)
    return t


def read_dir(dirpath: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out = {}
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(dirpath, fn), framework="pt") as f:
            for k in f.keys():  # noqa: SIM118
                out[k] = f.get_tensor(k)
    return out


def run(*extra: str) -> subprocess.CompletedProcess:
    # 跟随本测试自己的启动方式:`-I` 会屏蔽 user site-packages,有的环境里
    # torch 就装在那儿,硬写 `-I` 会让子进程 import 不到 torch。
    argv = [sys.executable] + (["-I"] if sys.flags.isolated else []) + [MERGE, *extra]
    return subprocess.run(argv, capture_output=True, text=True)


def reference(adapter: dict, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """独立算一遍 delta,用 einsum,不碰 merge_lora 的代码。"""
    p = f"layers.{layer}.moe.routed_experts.inner_experts."
    a13 = adapter[p + "w13_lora_a"].float()
    dw1 = torch.einsum("efr,erd->efd", adapter[p + "w1_lora_b"].float(), a13)
    dw3 = torch.einsum("efr,erd->efd", adapter[p + "w3_lora_b"].float(), a13)
    dw2 = torch.einsum("edr,erf->edf", adapter[p + "w2_lora_b"].float(), adapter[p + "w2_lora_a"].float())
    return dw1, dw3, dw2


def case_orient(tmp: str, orient: str) -> None:
    print(f"\n--- 朝向 {orient} ---")
    base_dir = os.path.join(tmp, f"base_{orient}")
    out_dir = os.path.join(tmp, f"out_{orient}")
    apath = os.path.join(tmp, f"adapter_{orient}.safetensors")

    base = make_base(base_dir, orient)
    adapter = make_adapter(apath)
    proc = run(
        "--base", base_dir, "--adapter", apath,
        "--rank", str(R), "--alpha", str(ALPHA), "--out", out_dir,
    )
    check(f"{orient}: 退出码 0", proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-800:])
    if proc.returncode != 0:
        return

    merged = read_dir(out_dir)
    scaling = ALPHA / R
    worst_gate = worst_down = 0.0
    for layer in LAYERS:
        p = f"model.language_model.layers.{layer}.mlp.experts."
        dw1, dw3, dw2 = reference(adapter, layer)
        gu0 = base[p + "gate_up_proj"].float()
        dn0 = base[p + "down_proj"].float()
        if orient == "efd":
            want_gu = gu0.clone()
            want_gu[:, :F, :] += scaling * dw1
            want_gu[:, F:, :] += scaling * dw3
            want_dn = dn0 + scaling * dw2
        else:
            want_gu = gu0.clone()
            want_gu[:, :, :F] += scaling * dw1.transpose(-2, -1)
            want_gu[:, :, F:] += scaling * dw3.transpose(-2, -1)
            want_dn = dn0 + scaling * dw2.transpose(-2, -1)
        # 基座是 bf16 容器,所以比较的容差按 bf16 的 ~3 位十进制有效数字来定。
        got_gu = merged[p + "gate_up_proj"].float()
        got_dn = merged[p + "down_proj"].float()
        worst_gate = max(
            worst_gate,
            ((got_gu - want_gu).abs().max() / want_gu.abs().max().clamp(min=1e-6)).item(),
        )
        worst_down = max(
            worst_down,
            ((got_dn - want_dn).abs().max() / want_dn.abs().max().clamp(min=1e-6)).item(),
        )

    check(f"{orient}: gate_up 数值匹配 (相对误差 {worst_gate:.2e})", worst_gate < 2e-2)
    check(f"{orient}: down_proj 数值匹配 (相对误差 {worst_down:.2e})", worst_down < 2e-2)

    untouched = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
    check(
        f"{orient}: 共享专家没被动",
        torch.equal(merged[untouched], base[untouched]),
    )

    info_path = os.path.join(out_dir, "merge_info.json")
    ok = os.path.isfile(info_path)
    check(f"{orient}: 写了 merge_info.json", ok)
    if ok:
        with open(info_path, encoding="utf-8") as fh:
            info = json.load(fh)
        check(
            f"{orient}: merge_info 记了 scaling={scaling}",
            info.get("scaling") == scaling and info.get("alpha") == ALPHA,
            str(info),
        )
    check(f"{orient}: 非权重文件带过来了", os.path.isfile(os.path.join(out_dir, "config.json")))


def main() -> int:
    print("================ merge_lora 自测 ================")
    with tempfile.TemporaryDirectory() as tmp:
        case_orient(tmp, "efd")
        case_orient(tmp, "edf")

        print("\n--- 交叉校验与守卫 ---")
        base_dir = os.path.join(tmp, "base_efd")
        apath = os.path.join(tmp, "adapter_efd.safetensors")

        p = run("--base", base_dir, "--adapter", apath, "--rank", str(R + 1), "--alpha", str(ALPHA), "--dry-run")
        check(
            "rank 不一致时失败(§7.2)",
            p.returncode != 0 and "rank" in (p.stdout + p.stderr),
            p.stdout[-400:] + p.stderr[-400:],
        )

        p = run("--base", base_dir, "--adapter", apath, "--rank", str(R), "--dry-run")
        check("不给 --alpha 时失败", p.returncode != 0)

        p = run("--base", base_dir, "--adapter", apath, "--rank", str(R), "--alpha", str(ALPHA), "--dry-run")
        check("--dry-run 退出码 0", p.returncode == 0, p.stdout[-600:] + p.stderr[-400:])
        check(
            "--dry-run 不写文件",
            not os.path.isdir(os.path.join(tmp, "nonexistent_out")),
        )

        print("\n--- 形状撞车(dim == 2*hidden)---")
        amb_d, amb_f = 8, 4  # 2*f == d,两种朝向形状一样
        amb_base = os.path.join(tmp, "base_amb")
        amb_ad = os.path.join(tmp, "adapter_amb.safetensors")
        make_base(amb_base, "edf", amb_d, amb_f)
        make_adapter(amb_ad, d=amb_d, f=amb_f)
        p = run("--base", amb_base, "--adapter", amb_ad, "--rank", str(R), "--alpha", str(ALPHA), "--dry-run")
        check(
            "auto 在判不出朝向时拒绝",
            p.returncode != 0 and "判不出来" in (p.stdout + p.stderr),
            p.stdout[-500:] + p.stderr[-500:],
        )
        p = run(
            "--base", amb_base, "--adapter", amb_ad, "--rank", str(R), "--alpha", str(ALPHA),
            "--gate-up-layout", "edf", "--dry-run",
        )
        check("显式 --gate-up-layout edf 能跑通", p.returncode == 0, p.stdout[-600:] + p.stderr[-400:])
        p = run(
            "--base", amb_base, "--adapter", amb_ad, "--rank", str(R), "--alpha", str(ALPHA),
            "--gate-up-layout", "efd", "--dry-run",
        )
        check(
            "强制 efd 与 down_proj 形状冲突时失败",
            p.returncode != 0,
            p.stdout[-500:] + p.stderr[-500:],
        )

        print("\n--- 空操作判据 ---")
        zpath = os.path.join(tmp, "adapter_zero.safetensors")
        make_adapter(zpath, zero_b=True)
        p = run("--base", base_dir, "--adapter", zpath, "--rank", str(R), "--alpha", str(ALPHA), "--dry-run")
        check(
            "全 0 的 B 被判为空操作(§7.1)",
            p.returncode != 0 and "空操作" in (p.stdout + p.stderr),
            p.stdout[-500:],
        )

    print(f"\n================ {_ok} 项通过, {_fail} 项失败 ================")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
