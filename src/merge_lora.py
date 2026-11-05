#!/usr/bin/env python
"""把 adapter 合并回假量化 bf16 checkpoint(工单 §7.1 / §7.2)。

    # 先干跑:只报朝向、scaling、预计改动量,不写文件
    python -I src/merge_lora.py --base <假量化ckpt> --adapter <adapter目录> \
        --rank 16 --alpha 32 --dry-run

    python -I src/merge_lora.py --base <假量化ckpt> --adapter <adapter目录> \
        --rank 16 --alpha 32 --out <合并后ckpt>

``W_merged = W_hat + (alpha/rank) · B · A``,其中 ``W_hat`` 是假量化后的权重。

**不再重新量化。** DSv4.1 部署时做的是
``quant_int8(dequant(W_nbit) + (alpha/rank)·B·A)`` —— 那个 int8 是因为它的运行时
是 int8,而 int8 远宽于 3bit,所以补偿基本没被吃掉。我们这边是假量化评测,
基座本来就是 bf16 容器,把合并结果再压回 3bit 等于把刚学到的补偿大部分扣回去。
位宽账仍然成立:3bit 基座 + bf16 低秩增量,和 DSv4.1 报的那条曲线同一种口径。

----

**§7.2 不是形式主义,而且情况比工单写的更糟。**

工单说"从 adapter 文件 metadata 读 alpha 并交叉校验"。实际读过
``adapter_save.py:54`` 之后:它写的 metadata 是
``{"step": ..., "rank": str(_rank())}`` —— 那个 ``rank`` 是**进程 rank**,
不是 LoRA rank。**alpha 根本没有被存进 adapter 文件。**

所以:

* LoRA rank 可以从张量形状反推(``w13_lora_a`` 的 dim 1),本脚本**强制**
  它和 ``--rank`` 一致,不一致就抛。
* alpha **无法**从 adapter 恢复,必须由你给,``--alpha`` 没有默认值。
  DSv4.1 踩的就是这个:``assemble_adapter.py`` 的 ``--alpha`` 默认 32.0,
  训练日志是 4.0,评测照 metadata 传了 32 —— 实际加上去的 delta 比训练时
  学到的放大了 8 倍,那批 3bit+LoRA 参照数全部不可用。
* 本脚本把实际用的 ``rank`` / ``alpha`` / ``scaling`` 写进输出目录的
  ``merge_info.json`` **和** 每个分片的 safetensors metadata,
  这样评测端不可能再和训练端分叉。

----

**gate_up_proj 的朝向是自动判的,不是假设的。**

``state_dict_adapter.py:272-284`` 期望 HF 是 ``[E, dim, 2*hidden]``,在最后一维
融合,``chunk(2, dim=-1)`` 之后转置成 TT 的 ``w1_EFD [E, hidden, dim]``。
但 2026-11-05 对官方权重扫描记下的实测形状是 ``[256, 1024, 2048]``
= ``[E, 2*hidden, dim]``,也就是在**第 1 维**融合 —— 和 adapter 期望的正好转置。

两者只可能有一个对,而 ``dim=2048 != 2*hidden=1024``,所以形状本身就能把
朝向认出来,不用猜。本脚本按实际形状判,并把判断结果打出来。

**顺带:这个矛盾如果是真的,工单 §0.4(用上游 flavor 加载真权重)会在
``w1_EFD`` 上报 shape mismatch 并直接失败** —— 响的,不是静默的。
那种情况下要改的是 state_dict_adapter,不是本脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# adapter 参数名,见 deepseek_v4/lora.py:181-185 的 grouped_lora_class。
ADAPTER_NAMES = ("w13_lora_a", "w1_lora_b", "w3_lora_b", "w2_lora_a", "w2_lora_b")

# adapter 的 FQN 形如 layers.{i}.moe.routed_experts.inner_experts.w13_lora_a
_ADAPTER_RE = re.compile(
    r"layers\.(\d+)\.moe\.routed_experts\.inner_experts\.(" + "|".join(ADAPTER_NAMES) + r")$"
)
# 基座的 HF 键,见 state_dict_adapter.py:72 和 fakequant_rtn.py 记的实测布局。
_GATE_UP_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.gate_up_proj(\.weight)?$")
_DOWN_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.down_proj(\.weight)?$")
_SHARD_RE = re.compile(r"adapter_r(\d+)_step(\d+)\.safetensors$")


def die(msg: str, *lines: str):
    print(f"\n!! {msg}", file=sys.stderr)
    for line in lines:
        print(f"   {line}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- adapter 读入
def _parse_placements(text: str | None) -> str:
    return (text or "").replace(" ", "")


def load_adapter(path: str, step: int | None) -> dict[str, torch.Tensor]:
    """读 adapter。单文件直接读;目录则按 rank 拼。

    拼接只支持**专家维(dim 0)分片**这一种情况 —— ep=8 / dp_shard=1 下就是它。
    其他 placement 直接拒绝并指向 DSv4.1 的 assemble_adapter.py,
    因为猜错拼接顺序会产出一个看着能用、数值全错的 adapter。
    """
    if os.path.isfile(path):
        out = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():  # noqa: SIM118
                out[k] = f.get_tensor(k)
        print(f"adapter: 单文件 {path},{len(out)} 个张量")
        return out

    if not os.path.isdir(path):
        die(f"{path} 既不是文件也不是目录")

    shards: dict[int, list[tuple[int, str]]] = {}
    for fn in sorted(os.listdir(path)):
        m = _SHARD_RE.search(fn)
        if m:
            shards.setdefault(int(m.group(2)), []).append((int(m.group(1)), fn))
    if not shards:
        die(
            f"{path} 下没有 adapter_r*_step*.safetensors",
            "如果 adapter 是单个已拼好的文件,直接把 --adapter 指到那个文件。",
        )

    use_step = step if step is not None else max(shards)
    if use_step not in shards:
        die(f"没有 step {use_step} 的分片", f"现有:{sorted(shards)}")
    files = sorted(shards[use_step])
    print(f"adapter: step {use_step},{len(files)} 个 rank 分片")

    per_rank: dict[int, dict[str, torch.Tensor]] = {}
    meta: dict[str, str] = {}
    for rk, fn in files:
        with safe_open(os.path.join(path, fn), framework="pt") as f:
            meta.update(f.metadata() or {})
            per_rank[rk] = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118

    keys = sorted(per_rank[files[0][0]])
    for rk, _ in files:
        if sorted(per_rank[rk]) != keys:
            die(f"rank {rk} 的键集合和 rank {files[0][0]} 不一致 —— 分片不配对")

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        parts = [per_rank[rk][k] for rk, _ in files]
        gs = meta.get(k + ".global_shape")
        if gs is None:
            if len(parts) == 1:
                out[k] = parts[0]
                continue
            die(
                f"{k} 没有 global_shape metadata,无法判断怎么拼",
                "用 DSv4.1 的 dsv4/unisvq/assemble_adapter.py 先拼好,再把单文件喂给本脚本。",
            )
        global_shape = [int(x) for x in gs.split(",")]
        placements = _parse_placements(meta.get(k + ".placements"))
        cat = torch.cat(parts, dim=0)
        if list(cat.shape) != global_shape:
            die(
                f"{k} 按 dim 0 拼出来是 {list(cat.shape)},但 global_shape 是 {global_shape}",
                f"placements = {placements or '<未记录>'}",
                "本脚本只支持专家维分片。其他情况用 DSv4.1 的 assemble_adapter.py。",
            )
        if placements and "Shard(dim=0)" not in placements and "Shard(0)" not in placements:
            die(
                f"{k} 的 placements 是 {placements},不是专家维分片",
                "形状碰巧对得上也不代表拼对了。用 assemble_adapter.py。",
            )
        out[k] = cat

    print(f"adapter: 拼成 {len(out)} 个全局张量")
    return out


def group_adapter(tensors: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    skipped = []
    for k, v in tensors.items():
        m = _ADAPTER_RE.search(k)
        if not m:
            if "lora_" in k:
                skipped.append(k)
            continue
        by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = v
    if skipped:
        die(
            f"有 {len(skipped)} 个 lora_ 张量不在路由专家上 —— 训练配置和本脚本不匹配。",
            f"例:{skipped[0]}",
            "qwen36_lora_recipe 用的是 target_modules=(),不该出现 dense adapter。",
        )
    if not by_layer:
        die("adapter 里没有任何路由专家 LoRA 张量", "确认训练确实挂上了 converter(工单 §0.1)。")
    for layer, d in sorted(by_layer.items()):
        missing = [n for n in ADAPTER_NAMES if n not in d]
        if missing:
            die(f"第 {layer} 层缺 {missing}")
    return by_layer


# ---------------------------------------------------------------- delta
@torch.no_grad()
def layer_deltas(d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 TT 朝向的 (dW1 [E,F,D], dW3 [E,F,D], dW2 [E,D,F])。"""
    a13 = d["w13_lora_a"].float()  # [E, R, D]
    b1 = d["w1_lora_b"].float()  # [E, F, R]
    b3 = d["w3_lora_b"].float()  # [E, F, R]
    a2 = d["w2_lora_a"].float()  # [E, R, F]
    b2 = d["w2_lora_b"].float()  # [E, D, R]
    return b1 @ a13, b3 @ a13, b2 @ a2


def infer_dims(by_layer: dict[int, dict[str, torch.Tensor]]) -> tuple[int, int, int, int]:
    d = by_layer[min(by_layer)]
    e, r, dim = d["w13_lora_a"].shape
    f = d["w1_lora_b"].shape[1]
    for name, want in (
        ("w1_lora_b", (e, f, r)),
        ("w3_lora_b", (e, f, r)),
        ("w2_lora_a", (e, r, f)),
        ("w2_lora_b", (e, dim, r)),
    ):
        if tuple(d[name].shape) != want:
            die(f"{name} 形状是 {tuple(d[name].shape)},按 E={e} R={r} D={dim} F={f} 应该是 {want}")
    return e, r, dim, f


# ---------------------------------------------------------------- 合并
@torch.no_grad()
def merge_into(
    base: torch.Tensor,
    dw_gate: torch.Tensor,
    dw_up: torch.Tensor | None,
    scaling: float,
    dim: int,
    hidden: int,
    key: str,
    layout: str = "auto",
) -> tuple[torch.Tensor, float, str]:
    """把 delta 加到一个基座张量上。朝向按形状判。

    ``layout`` 为 ``auto`` 时按形状判;``efd`` / ``edf`` 强制。
    两种朝向的形状相同时(``dim == 2*hidden``,或 down_proj 的 ``dim == hidden``)
    形状**判不出来**,此时 auto 会直接抛 —— 静默选一边就是在赌,
    而赌错了产出的 checkpoint 看着正常、数值全错。

    返回 (新张量, 改动元素比例, 朝向描述)。
    """
    shape = tuple(base.shape)
    out = base.float()

    if dw_up is not None:  # gate_up_proj,融合的
        if layout == "auto" and dim == 2 * hidden:
            die(
                f"{key}: dim ({dim}) == 2*hidden ({2 * hidden}),两种朝向的形状一样,判不出来。",
                "用 --gate-up-layout efd(即 [E, 2*hidden, dim],第 1 维融合)",
                "或 --gate-up-layout edf(即 [E, dim, 2*hidden],最后一维融合)显式指定。",
                "判据:看官方 checkpoint 里 gate_up_proj 的实际形状。",
            )
        if layout == "efd" or (layout == "auto" and shape[1:] == (2 * hidden, dim)):
            if shape[1:] != (2 * hidden, dim):
                die(f"{key}: 强制 efd 但形状是 {shape},应为 [E,{2 * hidden},{dim}]")
            orient = f"[E, 2*hidden, dim] = {shape},第 1 维融合"
            out[:, :hidden, :] += scaling * dw_gate
            out[:, hidden:, :] += scaling * dw_up
        elif layout == "edf" or shape[1:] == (dim, 2 * hidden):
            if shape[1:] != (dim, 2 * hidden):
                die(f"{key}: 强制 edf 但形状是 {shape},应为 [E,{dim},{2 * hidden}]")
            orient = f"[E, dim, 2*hidden] = {shape},最后一维融合(与 state_dict_adapter 一致)"
            out[:, :, :hidden] += scaling * dw_gate.transpose(-2, -1)
            out[:, :, hidden:] += scaling * dw_up.transpose(-2, -1)
        else:
            die(
                f"{key} 的形状 {shape} 既不是 [E,{2 * hidden},{dim}] 也不是 [E,{dim},{2 * hidden}]",
                "基座和 adapter 不是同一个模型,或者 dim/hidden 推错了。",
            )
    else:  # down_proj
        # 同一个 flag 管两个键:实测布局两边都是 [out, in]
        # (gate_up [E,2F,D] / down [E,D,F]),adapter 期望的布局两边都是 [in, out]。
        if layout == "auto" and dim == hidden:
            die(
                f"{key}: dim == hidden == {dim},两种朝向的形状一样,判不出来。",
                "用 --gate-up-layout efd 或 edf 显式指定。",
            )
        if layout == "efd" or (layout == "auto" and shape[1:] == (dim, hidden)):
            if shape[1:] != (dim, hidden):
                die(f"{key}: 强制 efd 但形状是 {shape},应为 [E,{dim},{hidden}]")
            orient = f"[E, dim, hidden] = {shape}"
            out += scaling * dw_gate
        elif layout == "edf" or shape[1:] == (hidden, dim):
            if shape[1:] != (hidden, dim):
                die(f"{key}: 强制 edf 但形状是 {shape},应为 [E,{hidden},{dim}]")
            orient = f"[E, hidden, dim] = {shape}(与 state_dict_adapter 一致)"
            out += scaling * dw_gate.transpose(-2, -1)
        else:
            die(f"{key} 的形状 {shape} 既不是 [E,{dim},{hidden}] 也不是 [E,{hidden},{dim}]")

    new = out.to(base.dtype)
    changed = (new != base).sum().item() / base.numel()
    return new, changed, orient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="假量化 bf16 checkpoint 目录")
    ap.add_argument("--adapter", required=True, help="adapter 目录或单个 safetensors")
    ap.add_argument("--rank", type=int, required=True, help="训练时的 LoRA rank,与张量形状交叉校验")
    ap.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="训练时的 alpha。**没有默认值** —— adapter 文件里没存它,"
        "给错了会静默放缩 delta,见本文件 docstring 里 DSv4.1 的 8 倍事故。",
    )
    ap.add_argument(
        "--gate-up-layout",
        choices=("auto", "efd", "edf"),
        default="auto",
        help="专家权重朝向。auto=按形状判(Qwen3.6 的 dim=2048 != 2*hidden=1024,判得出来);"
        "efd=[E, 2*hidden, dim] / [E, dim, hidden];edf=[E, dim, 2*hidden] / [E, hidden, dim]",
    )
    ap.add_argument("--step", type=int, help="用哪一步的分片(默认最新)")
    ap.add_argument("--out", help="输出目录;--dry-run 时可省")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        die("要么给 --out,要么加 --dry-run")

    by_layer = group_adapter(load_adapter(args.adapter, args.step))
    e, r, dim, hidden = infer_dims(by_layer)

    # §7.2 的交叉校验。rank 能从形状反推,所以这里是硬校验。
    if r != args.rank:
        die(
            f"adapter 张量推出 rank={r},但 --rank 给的是 {args.rank}",
            "不一致就停 —— scaling = alpha/rank,错了整批数不可比。",
        )
    scaling = args.alpha / args.rank

    print("\n================ 合并参数 ================")
    print(f"层数      {len(by_layer)}  (层号 {min(by_layer)}..{max(by_layer)})")
    print(f"E/R/D/F   {e} / {r} / {dim} / {hidden}")
    print(f"rank      {args.rank}   (与张量形状一致)")
    print(f"alpha     {args.alpha}   <- 来自命令行,adapter 里没有存")
    print(f"scaling   {scaling}")
    print("adapter 里不含 alpha,所以这个 scaling 的正确性只能由你保证:")
    print("务必和训练日志里的 alpha 对一遍。DSv4.1 就是在这里把 delta 放大了 8 倍。")

    shard_files = sorted(f for f in os.listdir(args.base) if f.endswith(".safetensors"))
    if not shard_files:
        die(f"{args.base} 下没有 .safetensors")

    if args.out and not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    print("\n================ 逐分片 ================")
    deltas = {layer: layer_deltas(d) for layer, d in by_layer.items()}
    touched: list[tuple[str, float]] = []
    orients: set[str] = set()
    unmatched_layers = set(by_layer)

    for fn in shard_files:
        src = os.path.join(args.base, fn)
        tensors: dict[str, torch.Tensor] = {}
        meta: dict[str, str] = {}
        n_hit = 0
        with safe_open(src, framework="pt") as f:
            meta.update(f.metadata() or {})
            for k in f.keys():  # noqa: SIM118
                t = f.get_tensor(k)
                mg, md = _GATE_UP_RE.search(k), _DOWN_RE.search(k)
                layer = int((mg or md).group(1)) if (mg or md) else None
                if layer is not None and layer in deltas:
                    dw1, dw3, dw2 = deltas[layer]
                    if mg:
                        t, changed, orient = merge_into(
                            t, dw1, dw3, scaling, dim, hidden, k, args.gate_up_layout
                        )
                    else:
                        t, changed, orient = merge_into(
                            t, dw2, None, scaling, dim, hidden, k, args.gate_up_layout
                        )
                    orients.add(orient)
                    touched.append((k, changed))
                    unmatched_layers.discard(layer)
                    n_hit += 1
                tensors[k] = t

        if n_hit:
            print(f"  {fn}: 合并了 {n_hit} 个张量")
        if args.out and not args.dry_run:
            meta["lora_rank"] = str(args.rank)
            meta["lora_alpha"] = str(args.alpha)
            meta["lora_scaling"] = str(scaling)
            meta.setdefault("format", "pt")
            save_file(tensors, os.path.join(args.out, fn), metadata=meta)

    print("\n================ 结果 ================")
    for orient in sorted(orients):
        print(f"朝向      {orient}")
    if unmatched_layers:
        die(
            f"adapter 有 {len(unmatched_layers)} 层在基座里找不到对应专家张量:"
            f"{sorted(unmatched_layers)[:8]}",
            "层号对不上说明基座和训练用的不是同一个模型。",
        )
    if not touched:
        die("一个张量都没合并 —— 基座里找不到 gate_up_proj / down_proj 键。")

    ratios = sorted(c for _, c in touched)
    lo, mid, hi = ratios[0], ratios[len(ratios) // 2], ratios[-1]
    print(f"合并张量  {len(touched)} 个")
    print(f"改动比例  min {lo:.4f}  p50 {mid:.4f}  max {hi:.4f}")
    # 工单 §7.1 的验收:千分之几说明 merge 是空操作。
    if hi < 0.01:
        print(
            "\n!! 最大改动比例不到 1% —— merge 基本是空操作。"
            "\n   要么 adapter 没训出东西(看训练 loss),要么 scaling 太小。"
        )
        return 1
    print("改动量正常(不是空操作)。")

    if args.dry_run:
        print("\n--dry-run:没有写任何文件。")
        return 0

    # 非权重文件原样带过去,并记下这次合并用的参数。
    for fn in os.listdir(args.base):
        if fn.endswith(".safetensors"):
            continue
        s = os.path.join(args.base, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, fn))
    with open(os.path.join(args.out, "merge_info.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "base": os.path.abspath(args.base),
                "adapter": os.path.abspath(args.adapter),
                "step": args.step,
                "rank": args.rank,
                "alpha": args.alpha,
                "scaling": scaling,
                "num_experts": e,
                "dim": dim,
                "moe_hidden_dim": hidden,
                "layers": sorted(by_layer),
                "gate_up_layout": args.gate_up_layout,
                "orientations": sorted(orients),
                "merged_tensors": len(touched),
                "changed_ratio": {"min": lo, "p50": mid, "max": hi},
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n写出 -> {args.out}")
    print("评测这个目录时 MODEL_PATH 指到它;scaling 已记在 merge_info.json 和分片 metadata 里。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
