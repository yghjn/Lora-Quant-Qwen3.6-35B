#!/usr/bin/env python
"""工单 §4.1:按校准集路由计数挑热专家(混合精度回退的选择依据)。

    # 挑 hot26(256 的 10%),顺便看位宽账
    python -I src/mp_select.py --in <ckpt> --acts acts/ --out keep_hot26.json

    # 只看位宽账,不读任何数据
    python -I src/mp_select.py --bits-table --experts 256 --bits 2

----

**和原版 `mp_select.py` 的唯一结构性区别:输入不同。**

原版读 `route41/route_calib_e*.json` —— 那是 DSv4.1 寄生在 vLLM 里的 collector
吐出来的路由计数。本路线没有那个 collector(工单 §C.2),所以这里直接用
**checkpoint 自带的 router 权重 + 校准激活**重算路由:

    logits = x @ router_w.T   ->   topk_ids   ->   逐专家计数

和 `offline_ptq.route()` 是同一个函数,所以选专家用的路由和量化时分 token 用的
路由**保证是同一套**。原版那个"读错 route 目录会按错的拓扑悄悄选出一份看起来
正常的 keep 列表"的风险类别因此不存在了 —— 但换来一个新的:激活来自错的模型。
所以硬校验改成查 router 形状、激活 dim、专家数三者一致,不符就立刻退出。

----

**位宽账:这是本文件最该被先读的部分。**

平均位宽 = `NB + (keep_bits - NB) * K / E`。`keep_bits` 是**保留专家的位宽**,
它不是细节,它决定这个产物能不能和 DSv4.1 的数比:

    DSv4.1 实测 "2bit + hot39 回退 = 2.609 bit, GPQA 65.66"
      -> 39/384 = 10.16%,keep_bits = 8(int8 容器)
      -> 2 + (8-2) * 0.1016 = 2.609   对得上

    本路线全程 bf16(工单 §C.1),所以 keep_bits = 16:
      -> hot26 = 2 + (16-2) * 0.1016 = 3.422 bit

**3.422 不是 2.609。** 拿 hot26-bf16 的分数去和 DSv4.1 的 65.66 比,是拿高了
0.81 bit 的模型去比 —— 赢了不说明方法更好。同位宽预算下的等价保留个数是

    K = round(E * (2.609 - 2) / (16 - 2)) = 11

所以三个可比的点:hot11-bf16(2.609 bit,和 DSv4.1 同预算)、
hot26-bf16(3.422 bit,保留比例和 DSv4.1 相同但更贵)、
hot26-int8(2.609 bit,需要 int8 容器,`core.requant_int8_perchannel` 一行调用,
但那是工单 §C.1 明确不做的事)。本脚本把三个都算出来打印,选哪个是人的决定。

产物尺寸同理(路由专家 32.212 G 参数):

    2bit 全量          8.05 GB
    + hot26 保 int8   10.50 GB
    + hot26 保 bf16   13.77 GB   <- 几乎翻倍,这也是 DSv4.1 当初上 int8 的原因

----

**两个变体。** `hot` = 校准 token 最多的 K 个(承载最多计算量);
`cold` = 最少的 K 个(Hessian 样本最少、2bit 误差最大,长尾知识可能在这里)。
原版两个都出,这里照办 —— cold 是个便宜的对照假设,不是主路径。

并列打破按 `(-count, expert_id)`,确定性的。**零计数的专家之间排序没有意义**,
所以如果 K 大于非零计数的专家数,会警告:那时 hot 列表的尾部是任意的。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_ptq as ptq  # noqa: E402

# 路由专家总参数量(Qwen3.6-35B-A3B):40 层 x 256 专家 x 3 x 2048x512
ROUTED_PARAMS = 40 * 256 * 3 * 2048 * 512
DSV41_K, DSV41_E, DSV41_KEEP_BITS = 39, 384, 8
# DSv4.1 实测的那个点:2bit 基座 + hot39 保 int8 = 2.609 bit, GPQA 65.66。
# **它只对 2bit 基座成立**,3bit 基座下 DSv4.1 没有混精的公开数。
DSV41_BASE_BITS, DSV41_GPQA = 2, 65.66


def _pad(s: str, width: int) -> str:
    """按显示宽度左对齐。中日韩字符占两列,`str.ljust` 按码点算会把表排歪。"""
    import unicodedata

    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(width - w, 0)


def avg_bits(nbits: float, keep_bits: float, k: int, e: int) -> float:
    """平均位宽。保留 k 个专家在 keep_bits,其余 nbits。"""
    return nbits + (keep_bits - nbits) * k / e


def equiv_k(k_src: int, e_src: int, e: int) -> int:
    """按**保留比例相同**换算到本模型的保留个数。"""
    return round(k_src * e / e_src)


def k_for_bits(target: float, nbits: float, keep_bits: float, e: int) -> int:
    """按**平均位宽相同**反解保留个数。"""
    if keep_bits == nbits:
        return 0
    return round(e * (target - nbits) / (keep_bits - nbits))


def product_gb(nbits: float, keep_bits: float, k: int, e: int,
               params: int = ROUTED_PARAMS) -> float:
    return params * avg_bits(nbits, keep_bits, k, e) / 8 / 1e9


def print_bits_table(e: int, nbits: int, k_user: int | None = None) -> None:
    """位宽账。不读任何数据,所以可以单独跑(--bits-table)。"""
    dsv_bits = avg_bits(nbits, DSV41_KEEP_BITS, DSV41_K, DSV41_E)
    k_ratio = equiv_k(DSV41_K, DSV41_E, e)
    k_budget_16 = k_for_bits(dsv_bits, nbits, 16, e)
    # ours=False 的那行是 DSv4.1 自己的模型,参数量不同,所以不报产物尺寸
    rows = [
        (f"DSv4.1 hot{DSV41_K}/{DSV41_E} 保 int8", DSV41_K, DSV41_E, 8, False),
        (f"同保留比例 hot{k_ratio} 保 bf16", k_ratio, e, 16, True),
        (f"同保留比例 hot{k_ratio} 保 int8", k_ratio, e, 8, True),
        (f"同位宽预算 hot{k_budget_16} 保 bf16", k_budget_16, e, 16, True),
    ]
    if k_user is not None and k_user not in (k_ratio, k_budget_16):
        rows.append((f"本次选的 hot{k_user} 保 bf16", k_user, e, 16, True))
        rows.append((f"本次选的 hot{k_user} 保 int8", k_user, e, 8, True))
    print(f"\n位宽账(基座 {nbits} bit,E={e},路由专家 {ROUTED_PARAMS / 1e9:.2f} G 参数)")
    print(f"  {_pad('配置', 32)}{'保留比':>8} {'平均位宽':>10} {'产物':>10}")
    for name, k, ee, kb, ours in rows:
        size = f"{product_gb(nbits, kb, k, ee):>7.2f} GB" if ours else "      —  "
        print(f"  {_pad(name, 32)}{100.0 * k / ee:>7.2f}% {avg_bits(nbits, kb, k, ee):>9.3f} bit {size}")
    gap = avg_bits(nbits, 16, k_ratio, e) - dsv_bits
    if nbits == DSV41_BASE_BITS:
        print(f"  DSv4.1 实测的 {DSV41_GPQA} GPQA 就是第一行那个点({dsv_bits:.3f} bit)。")
        print(f"  要和它比,用'同位宽预算'那行;用'同保留比例 + bf16'那行会多 {gap:.2f} bit 的便宜。")
    else:
        print(f"  !! DSv4.1 的混精实测数({DSV41_GPQA} GPQA)只有 "
              f"{DSV41_BASE_BITS}bit 基座那一个点,它是 "
              f"{avg_bits(DSV41_BASE_BITS, DSV41_KEEP_BITS, DSV41_K, DSV41_E):.3f} bit。")
        print(f"     第一行是把它的保留比例套到 {nbits}bit 基座上算出来的**推算值**,"
              "没有对应的实测分数。")
    if k_budget_16 > 0:
        got = avg_bits(nbits, 16, k_budget_16, e)
        print(f"  (hot{k_budget_16} 实际是 {got:.3f} bit,比 {dsv_bits:.3f} 低 "
              f"{dsv_bits - got:.3f} —— K 取整只能取到整数个专家,偏保守那一侧。)")


# ---------------------------------------------------------------- 路由计数
@torch.no_grad()
def route_counts(src: str, acts: str, layers: list[int], idx: dict,
                 dim: int, layout: str, top_k: int) -> dict[int, torch.Tensor]:
    """逐层数每个专家吃到多少 token。返回 {层号: [E] 的 long 张量}。

    `topk` 返回的 ids 在一行内不重复,所以 bincount 的计数就是"命中的 token 数",
    和 `expert_hessians` 里 `(ids == e).any(-1)` 的口径一致。
    """
    out: dict[int, torch.Tensor] = {}
    for ln in layers:
        d = idx[ln]
        if "router" not in d:
            ptq.die(f"第 {ln} 层没有 router 权重",
                    "选热专家必须靠 router 分 token,没有 router 就没有依据。")
        router_w = ptq.get_tensor(src, d["router"][0], d["router"][1])
        e_n = d["gate_up"][2][0]
        if tuple(router_w.shape) != (e_n, dim):
            ptq.die(f"第 {ln} 层 router 形状 {tuple(router_w.shape)},期望 ({e_n}, {dim})",
                    "激活或 checkpoint 对不上 —— 不要继续,选出来的专家号会是错的。")
        x = ptq.load_acts(acts, ln, dim)
        ids = ptq.route(x, router_w, top_k)
        cnt = torch.bincount(ids.reshape(-1), minlength=e_n)
        if cnt.numel() != e_n:
            ptq.die(f"第 {ln} 层数出 {cnt.numel()} 个专家,权重里是 {e_n} 个")
        out[ln] = cnt
        nz = int((cnt > 0).sum())
        print(f"[层 {ln:>2}] {x.shape[0]:>7} token -> {int(cnt.sum())} 次命中;"
              f"非零专家 {nz}/{e_n};计数 中位 {int(cnt.median())} / 最大 {int(cnt.max())}")
    return out


def select(cnt: dict[int, torch.Tensor], k: int, variant: str) -> dict[str, list[int]]:
    """{层号: [保留的专家号]}。并列按 (-count, expert_id),确定性。"""
    table: dict[str, list[int]] = {}
    for ln in sorted(cnt):
        c = cnt[ln]
        order = sorted(range(c.numel()), key=lambda e: (-int(c[e]), e))
        table[str(ln)] = sorted(order[:k] if variant == "hot" else order[-k:])
    return table


def report(cnt: dict[int, torch.Tensor], table: dict[str, list[int]], k: int,
           variant: str, rank_need: int, e_n: int) -> dict:
    """覆盖面 / 承载量 / 秩亏自检。返回写进 info 文件的统计。"""
    tot = kept_tok = 0
    thin = 0
    zero_tail = 0
    for ln_s, keep in table.items():
        c = cnt[int(ln_s)]
        tot += int(c.sum())
        kept_tok += sum(int(c[e]) for e in keep)
        thin += sum(1 for e in keep if int(c[e]) < rank_need)
        zero_tail += sum(1 for e in keep if int(c[e]) == 0)
    share = 100.0 * kept_tok / max(tot, 1)
    n_keep = k * len(table)
    print(f"\n{variant} 变体:每层保留 {k} 个,共 {n_keep} 个专家槽")
    print(f"  承载校准 token 的 {share:.1f}%(均匀路由下应为 {100.0 * k / e_n:.1f}%)")
    print(f"  其中 token < {rank_need}(w13 的 H 不满秩)的 {thin}/{n_keep} 个")
    if zero_tail:
        print(f"  !! 其中 {zero_tail} 个专家校准 token 为 0 —— 零计数之间排序没有意义,"
              f"这部分列表是任意的。减小 -k,或者加校准数据。")
    return {"variant": variant, "keep_per_layer": k, "layers": len(table),
            "calib_token_share_pct": round(share, 3),
            "experts_below_rank_need": thin, "experts_zero_token": zero_tail}


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", help="源 HF checkpoint 目录")
    ap.add_argument("--acts", help="各层 MoE 输入激活目录,每层一个 layer<NN>.pt")
    ap.add_argument("--out", help="输出的 keep json(hot);cold 变体写成 <stem>_cold.json")
    ap.add_argument("--layers", help="参与统计的层,默认 all")
    ap.add_argument("-k", "--keep", type=int,
                    help="每层保留几个专家。默认 = DSv4.1 hot39/384 的等比例换算")
    ap.add_argument("--variant", choices=("hot", "cold", "both"), default="both")
    ap.add_argument("--layout", choices=("auto", "efd", "edf"), default="auto")
    ap.add_argument("--top-k", type=int, help="router top-k;默认从 config.json 读")
    ap.add_argument("--bits", type=int, default=2, help="基座位宽,只用于位宽账")
    ap.add_argument("--bits-table", action="store_true",
                    help="只打位宽账然后退出,不读 checkpoint / 激活")
    ap.add_argument("--experts", type=int, help="配 --bits-table 用的专家数")
    args = ap.parse_args()

    if args.bits_table:
        print_bits_table(args.experts or 256, args.bits, args.keep)
        return 0

    for need, flag in ((args.src, "--in"), (args.acts, "--acts"), (args.out, "--out")):
        if not need:
            ptq.die(f"缺 {flag}", "或者用 --bits-table 只看位宽账。")

    idx = ptq.index_checkpoint(args.src)
    geo = ptq.read_geometry(args.src)
    layers = ptq.parse_layers(args.layers, list(idx))
    probe = idx[layers[0]]
    gu_shape, dn_shape = probe["gate_up"][2], probe["down"][2]
    e_n = gu_shape[0]
    if geo["num_experts"] not in (None, e_n):
        ptq.die(f"config.json 说 num_experts={geo['num_experts']},张量第 0 维是 {e_n}")
    a, b = gu_shape[1], gu_shape[2]
    dim = hidden = None
    for d_c, h_c in ((b, a // 2), (a, b // 2)):
        if sorted(dn_shape[1:]) == sorted([d_c, h_c]):
            dim, hidden = d_c, h_c
            break
    if dim is None:
        ptq.die(f"gate_up {gu_shape} 和 down {dn_shape} 推不出一致的 (dim, hidden)")
    layout = ptq.decide_layout(gu_shape, dn_shape, dim, hidden, args.layout)
    top_k = args.top_k or geo["top_k"]
    if not top_k:
        ptq.die("推不出 top_k", "config.json 里没有 num_experts_per_tok,用 --top-k 给。")

    k = args.keep or equiv_k(DSV41_K, DSV41_E, e_n)
    if not 0 < k < e_n:
        ptq.die(f"-k 必须在 (0, {e_n}) 内,给的是 {k}")

    print("================ 配置 ================")
    print(f"E / dim / hidden   {e_n} / {dim} / {hidden}")
    print(f"朝向 / top_k       {layout} / {top_k}")
    print(f"层                 {len(layers)}/{len(idx)} 层")
    print(f"每层保留           {k} 个({100.0 * k / e_n:.2f}%)"
          + ("" if args.keep else f"  <- 默认值,DSv4.1 hot{DSV41_K}/{DSV41_E} 的等比例换算"))
    if len(layers) < len(idx):
        print(f"\n!! 只统计了 {len(layers)}/{len(idx)} 层。没统计到的层在 "
              f"offline_ptq 里会**整层走 {args.bits}bit**,而且不会再提示一次。")

    print("\n================ 路由计数 ================")
    cnt = route_counts(args.src, args.acts, layers, idx, dim, layout, top_k)

    stem, ext = os.path.splitext(args.out)
    ext = ext or ".json"
    variants = ("hot", "cold") if args.variant == "both" else (args.variant,)
    infos = []
    for v in variants:
        table = select(cnt, k, v)
        infos.append(report(cnt, table, k, v, dim, e_n))
        # both:hot 写 --out,cold 写 <stem>_cold<ext>。单变体就都写 --out。
        path = args.out if args.variant != "both" or v == "hot" else f"{stem}_cold{ext}"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(table, fh)
        print(f"  -> {path}")

    print_bits_table(e_n, args.bits, k)

    info = {
        "source": os.path.abspath(args.src),
        "acts": os.path.abspath(args.acts),
        "num_experts": e_n, "dim": dim, "moe_hidden_dim": hidden,
        "top_k": top_k, "layout": layout,
        "layers_counted": layers, "layers_in_checkpoint": sorted(idx),
        "base_bits": args.bits,
        "avg_bits_keep_bf16": round(avg_bits(args.bits, 16, k, e_n), 4),
        "avg_bits_keep_int8": round(avg_bits(args.bits, 8, k, e_n), 4),
        "dsv41_reference_bits": round(avg_bits(args.bits, DSV41_KEEP_BITS, DSV41_K, DSV41_E), 4),
        "k_same_ratio_as_dsv41": equiv_k(DSV41_K, DSV41_E, e_n),
        "k_same_bits_as_dsv41_bf16": k_for_bits(
            avg_bits(args.bits, DSV41_KEEP_BITS, DSV41_K, DSV41_E), args.bits, 16, e_n),
        "variants": infos,
        "tie_break": "(-count, expert_id)",
    }
    ipath = f"{stem}_info.json"
    with open(ipath, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    print(f"  -> {ipath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
