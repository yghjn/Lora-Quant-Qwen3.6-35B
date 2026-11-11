#!/usr/bin/env python
"""工单 §2.2:离线 PTQ(RHT + 码本 + LDLQ),不寄生推理引擎。

    # 单层 dry-run(工单 §2.3,不能省)
    python -I src/offline_ptq.py run --in <ckpt> --out <dst> --bits 3 \
        --acts acts/ --layers 20

    # 没有校准数据时:H=I,即"没有校准的 QuIP"
    python -I src/offline_ptq.py run --in <ckpt> --out <dst> --bits 2 \
        --hessian identity

    # 只算 Hessian 存盘(给复用 / 调参)
    python -I src/offline_ptq.py hessian --in <ckpt> --acts acts/ --out hess/ --layers 20

    # 混合精度回退(工单 §4.2):先用 mp_select.py 挑热专家,再量化时跳过它们
    python -I src/mp_select.py --in <ckpt> --acts acts/ --out keep_hot26.json
    python -I src/offline_ptq.py run --in <ckpt> --out <dst> --bits 2 \
        --acts acts/ --keep-hot keep_hot26.json

----

**边界划在哪里,以及为什么。**

工单 §2.2 写的是"hook MoE block 输入 x 和 router 的 topk_ids"。实际只需要
前者:**router 本身就是 checkpoint 里的一个线性层**,拿到 MoE 输入的 hidden
states 之后,topk_ids、每专家的 xe、`a = swiglu(xe @ W13)`、两个 Hessian,
全都能用纯 PyTorch 在 CPU 上重算出来。

所以要卡要模型的部分被压到只剩一件事:

    **抓各层 MoE 输入的 hidden states,每层存一个 [N, dim] 的张量。**

那是约 40 行 forward hook,本文件**不含**它 —— 它依赖具体用哪个框架跑前向
(torchtitan / HF),而那个环境还没建起来,写了也验不了。本文件消费它的产物:
`--acts <目录>`,每层一个 `layer<NN>.pt`,内容是 `[N, dim]` 的 bf16/fp32 张量。
`--acts` 的格式校验在 `load_acts` 里,形状和 dtype 不对会直接抛。

剩下的全部可测,`verify/test_offline_ptq_cpu.py` 覆盖。

----

**和 DSv4.1 `collect()` 的三处不同**(工单 §2.2 脚注):

1. 去掉对 `xe` 和 `a` 的 `npu_dynamic_quant` —— 基座是 bf16,算子真正吃到的
   就是 `xe` 本身。这不是简化而是正确性:Hessian 必须反映算子真正吃到的分布。
2. `swiglu_limit` 为 0 时不 clamp。Qwen3.6 没有这个参数,所以不 clamp。
3. EP=1,全局专家号就是 `0..num_experts-1`。**`_signs` 的确定性完全依赖这个
   编号方案**,所以种子方案写进 manifest(`sign_seed_scheme`),
   任何重算必须用同一套。

----

**两处朝向/顺序,都做成显式而不是假设。**

* `--layout`:专家权重是 `[E, 2F, D]`(对官方权重实测到的)还是 `[E, D, 2F]`
  (`state_dict_adapter` 期望的)。auto 按形状判,`D == 2F` 时拒绝。
  和 `merge_lora.py` 同一套判法。
* `--gate-up-order`:融合的是 `[gate; up]` 还是 `[up; gate]`。
  **这里比合并那边更要紧**:`a = silu(gate) * up`,顺序弄反会让 `a` 整个变样,
  于是 `H2` 是错的,w2 的量化跟着错 —— 而 SNR 看着仍然正常,
  因为 SNR 只看 `||W - Ŵ||`,不看 H。这是会静默产出垃圾的那一类。
  判据:拿一条校准数据,两种顺序各算一次 logits,和 bf16 基座对齐的那个是对的;
  或者直接看官方的 modeling 代码。本脚本把实际用的顺序记进 manifest。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unisvq_core as core  # noqa: E402

# 键名。和 fakequant_rtn.py 保持一致。
_GATE_UP_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.gate_up_proj(\.weight)?$")
_DOWN_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.down_proj(\.weight)?$")
# router:两种命名都见过,所以都认
_ROUTER_RE = re.compile(r"layers\.(\d+)\.mlp\.(gate|router)\.weight$")
_ACT_RE = re.compile(r"layer0*(\d+)\.pt$")


def die(msg: str, *lines: str):
    print(f"\n!! {msg}", file=sys.stderr)
    for line in lines:
        print(f"   {line}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- 种子
def sign_seed(layer: int, expert: int, tag: str) -> int:
    """`rand_signs` 的种子。**方案一旦定下就不能改** —— 改了等于换了一套 RHT,
    已有产物全部作废。量化侧和任何重算侧必须用同一个函数。

    方案(记进 manifest 的 `sign_seed_scheme`):
        (layer * 100003 + expert * 17 + TAG_ID[tag]) % 2**31
    100003 是质数,远大于专家数 256,所以 (layer, expert) 不会撞。
    """
    tag_id = {"w13_in": 0, "w13_out": 1, "w2_in": 2, "w2_out": 3}[tag]
    return (layer * 100003 + expert * 17 + tag_id) % (2**31)


SIGN_SEED_SCHEME = "(layer*100003 + expert*17 + tag_id) % 2**31; tag_id: w13_in=0 w13_out=1 w2_in=2 w2_out=3"


# ---------------------------------------------------------------- 几何
def read_geometry(src: str) -> dict:
    """从 config.json 读 E / top_k,再用张量形状交叉校验。"""
    cfg_path = os.path.join(src, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cfg = {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def pick(*names):
        for d in (text, cfg):
            for n in names:
                if isinstance(d.get(n), int):
                    return d[n]
        return None

    return {
        "num_experts": pick("num_experts", "n_routed_experts", "moe_num_experts"),
        "top_k": pick("num_experts_per_tok", "moe_top_k", "top_k"),
    }


def decide_layout(gu_shape, dn_shape, dim: int, hidden: int, layout: str) -> str:
    """返回 'efd' 或 'edf'。和 merge_lora.py 同一套判法。"""
    if layout != "auto":
        return layout
    if dim == 2 * hidden:
        die(
            f"dim ({dim}) == 2*hidden ({2 * hidden}),两种朝向形状一样,判不出来。",
            "用 --layout efd 或 --layout edf 显式指定。",
        )
    if tuple(gu_shape[1:]) == (2 * hidden, dim) and tuple(dn_shape[1:]) == (dim, hidden):
        return "efd"
    if tuple(gu_shape[1:]) == (dim, 2 * hidden) and tuple(dn_shape[1:]) == (hidden, dim):
        return "edf"
    die(
        f"朝向认不出来:gate_up {tuple(gu_shape)} / down {tuple(dn_shape)}",
        f"按 dim={dim} hidden={hidden},应该是 [E,{2 * hidden},{dim}]/[E,{dim},{hidden}](efd)",
        f"或 [E,{dim},{2 * hidden}]/[E,{hidden},{dim}](edf)",
    )
    raise AssertionError("unreachable")


def as_out_in(t: torch.Tensor, layout: str) -> torch.Tensor:
    """把一个专家矩阵摆成 [out, in] —— `unisvq_qdq` 要的形状。"""
    return t if layout == "efd" else t.transpose(-2, -1)


def from_out_in(t: torch.Tensor, layout: str) -> torch.Tensor:
    return t if layout == "efd" else t.transpose(-2, -1)


# ---------------------------------------------------------------- 层索引
def index_checkpoint(src: str) -> dict:
    """扫一遍 header,建 层号 -> {键: (分片, 形状)} 的索引。只读 header。"""
    layers: dict[int, dict] = {}
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".safetensors")):
        with safe_open(os.path.join(src, fn), framework="pt") as f:
            for k in f.keys():  # noqa: SIM118
                for rx, role in ((_GATE_UP_RE, "gate_up"), (_DOWN_RE, "down"), (_ROUTER_RE, "router")):
                    m = rx.search(k)
                    if m:
                        layers.setdefault(int(m.group(1)), {})[role] = (
                            fn,
                            k,
                            list(f.get_slice(k).get_shape()),
                        )
                        break
    complete = {ln: d for ln, d in layers.items() if "gate_up" in d and "down" in d}
    if not complete:
        die(f"{src} 里没找到 mlp.experts.gate_up_proj / down_proj", "先跑 fakequant_rtn.py --scan 看键名。")
    return complete


def get_tensor(src: str, shard: str, key: str) -> torch.Tensor:
    with safe_open(os.path.join(src, shard), framework="pt") as f:
        return f.get_tensor(key)


# ---------------------------------------------------------------- Hessian
@torch.no_grad()
def route(x: torch.Tensor, router_w: torch.Tensor, top_k: int) -> torch.Tensor:
    """返回 topk_ids [N, top_k]。router 是个线性层:logits = x @ router_w.T。

    只要 ids,所以 softmax / sigmoid / 归一化方式都不影响结果 ——
    topk 只看 logits 的序。这是本脚本不需要知道 router 具体是哪种
    归一化的原因。
    """
    logits = x.float() @ router_w.float().T  # [N, E]
    return torch.topk(logits, k=top_k, dim=-1).indices


@torch.no_grad()
def expert_hessians(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w13: torch.Tensor,
    top_k: int,
    gate_first: bool,
    max_tok: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """逐专家累 H13 = xe^T xe 和 H2 = a^T a。

    x:       [N, dim]   该层 MoE 的输入 hidden states
    router_w:[E, dim]
    w13:     [E, 2*hidden, dim]  已摆成 [out, in]
    返回 (H13 [E,dim,dim], H2 [E,hidden,hidden], ntok [E])

    `a = silu(gate) * up`,**不 clamp**(Qwen3.6 没有 swiglu_limit)。
    fp32 累加:fp64 会把 [256,2048,2048] 的显存翻倍到 8.6 GiB/层,
    而 Hessian 之后还要 damp,精度不是瓶颈。
    """
    e_n, two_f, dim = w13.shape
    hidden = two_f // 2
    if x.shape[-1] != dim:
        die(f"激活的 dim 是 {x.shape[-1]},权重的是 {dim}")
    if router_w.shape != (e_n, dim):
        die(f"router 形状 {tuple(router_w.shape)},期望 ({e_n}, {dim})")

    ids = route(x, router_w, top_k)  # [N, top_k]
    H13 = torch.zeros(e_n, dim, dim, dtype=torch.float32)
    H2 = torch.zeros(e_n, hidden, hidden, dtype=torch.float32)
    ntok = torch.zeros(e_n, dtype=torch.long)

    xf = x.float()
    for e in range(e_n):
        sel = (ids == e).any(dim=-1).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        if max_tok and sel.numel() > max_tok:
            # 采样而不是截断:截断会偏向校准集前面的样本。
            perm = torch.randperm(sel.numel(), generator=torch.Generator().manual_seed(e))
            sel = sel[perm[:max_tok]]
        xe = xf[sel]  # [n, dim]
        H13[e] = xe.T @ xe
        g_u = xe @ w13[e].float().T  # [n, 2*hidden]
        if gate_first:
            gate, up = g_u[:, :hidden], g_u[:, hidden:]
        else:
            up, gate = g_u[:, :hidden], g_u[:, hidden:]
        a = torch.nn.functional.silu(gate) * up
        H2[e] = a.T @ a
        ntok[e] = sel.numel()
    return H13, H2, ntok


# ---------------------------------------------------------------- 混合精度
def load_keep_hot(path: str, e_n: int, layers: list[int]) -> dict[int, set[int]]:
    """读 `mp_select.py` 产出的 keep json:`{"<层号>": [专家号, ...]}`。

    工单 §4.2。被列进来的专家**不量化**,原权重直通,所以它们保持 bf16 ——
    平均位宽因此是 `NB + (16 - NB) * K / E`,不是 DSv4.1 那个 int8 容器的
    `NB + (8 - NB) * K / E`。两者差 0.8 bit,见 `mp_select.py` 的位宽账。

    三类静默失败,这里全部挡掉:
      * 专家号 >= E:keep 列表来自别的拓扑,会保错专家。
      * 要量化的层不在 keep 里:那一层**整层**走低位宽而不自知
        —— 原版 `mp_select.py` 把这一条列为"缺层"自检,照搬。
      * 某层保留了全部专家:等于那层没量化,但产物看着正常。
    """
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        die(f"读不了 keep json: {path}", str(exc))
    if not isinstance(raw, dict) or not raw:
        die(f"{path} 不是 {{层号: [专家号]}} 的非空字典", "用 src/mp_select.py 生成。")

    keep: dict[int, set[int]] = {}
    for k, v in raw.items():
        try:
            ln = int(k)
        except (TypeError, ValueError):
            die(f"{path} 的键 {k!r} 不是层号", "格式是 {\"20\": [1, 5, ...]};info 文件请另存。")
        if not isinstance(v, list) or any(not isinstance(e, int) for e in v):
            die(f"{path} 第 {ln} 层的值不是整数列表")
        bad = [e for e in v if not 0 <= e < e_n]
        if bad:
            die(f"{path} 第 {ln} 层有越界专家号 {bad[:5]}(E={e_n})",
                "这份 keep 列表来自别的拓扑 —— 继续下去会保错专家。")
        if len(set(v)) == e_n:
            die(f"{path} 第 {ln} 层保留了全部 {e_n} 个专家", "那层等于没量化。")
        keep[ln] = set(v)

    miss = [ln for ln in layers if ln not in keep]
    if miss:
        print(f"\n!! keep json 里没有这些要量化的层:{miss}")
        print(f"   它们会**整层**走低位宽。如果是有意的就忽略,否则先补全 "
              f"{os.path.basename(path)}。")
    extra = sorted(set(keep) - set(layers))
    if extra:
        print(f"   (keep json 里有 {len(extra)} 层本次不量化,已忽略:{extra[:8]})")
    return keep


def avg_bits_report(bits: int, keep: dict[int, set[int]] | None,
                    layers: list[int], e_n: int) -> dict:
    """按**本次量化的层**算平均位宽。保留的专家是 bf16 = 16 bit。"""
    slots = len(layers) * e_n
    kept = sum(len(keep.get(ln, ())) for ln in layers) if keep else 0
    return {
        "kept_experts": kept,
        "expert_slots": slots,
        "kept_ratio": round(kept / slots, 6) if slots else 0.0,
        "keep_bits": 16,
        "avg_bits": round(bits + (16 - bits) * kept / slots, 4) if slots else float(bits),
    }


def load_acts(acts_dir: str, layer: int, dim: int) -> torch.Tensor:
    path = None
    for fn in os.listdir(acts_dir):
        m = _ACT_RE.search(fn)
        if m and int(m.group(1)) == layer:
            path = os.path.join(acts_dir, fn)
            break
    if path is None:
        die(
            f"{acts_dir} 下没有第 {layer} 层的激活文件",
            "命名应为 layer<NN>.pt(例 layer20.pt),内容是 [N, dim] 的张量。",
            "抓 hidden states 的那段 hook 不在本仓库 —— 见本文件 docstring。",
        )
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, dict):
        t = t.get("x", t.get("hidden_states"))
    if not isinstance(t, torch.Tensor):
        die(f"{path} 里不是张量", "支持裸张量,或含 'x' / 'hidden_states' 键的 dict。")
    t = t.reshape(-1, t.shape[-1])
    if t.shape[-1] != dim:
        die(f"{path} 的最后一维是 {t.shape[-1]},权重要求 {dim}")
    return t


# ---------------------------------------------------------------- 量化一层
@torch.no_grad()
def quantize_layer(
    w13: torch.Tensor,
    w2: torch.Tensor,
    H13: torch.Tensor | None,
    H2: torch.Tensor | None,
    ntok: torch.Tensor | None,
    layer: int,
    bits: int,
    damp: float,
    use_ldlq: bool,
    keep: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """w13 [E,2F,D] / w2 [E,D,F],都是 [out,in]。返回量化后的两个张量 + 每专家记录。

    `keep` 里的专家**原权重直通**(工单 §4.2 的混合精度回退),记 `kept: True`
    且**不记 SNR** —— 未量化矩阵的 SNR 是 +inf,混进统计会把 p50 推到无意义。
    """
    e_n, two_f, dim = w13.shape
    hidden = two_f // 2
    Hd = core.hadamard(core.HAD_BLOCK)
    cb = core.LinearCodebook(core.D_GROUP, bits=bits)
    out13 = torch.empty_like(w13)
    out2 = torch.empty_like(w2)
    recs: list[dict] = []

    for e in range(e_n):
        rec: dict = {"expert": e}
        if keep and e in keep:
            out13[e] = w13[e]
            out2[e] = w2[e]
            rec["kept"] = True
            if ntok is not None:
                rec["ntok"] = int(ntok[e])
            recs.append(rec)
            continue
        for tag, W, H, in_n, out_n in (
            ("w13", w13[e], None if H13 is None else H13[e], dim, two_f),
            ("w2", w2[e], None if H2 is None else H2[e], hidden, dim),
        ):
            Wf = W.float()
            degraded = False
            if H is None:
                Hm = torch.eye(in_n)
                degraded = ntok is not None  # 有校准但这个专家没 token
            elif ntok is not None and int(ntok[e]) == 0:
                # 工单 §2.4 的验收点:ntok=0 必须被记成 DEGRADED,不能静默用零 Hessian。
                # 零 Hessian 会让 LDLQ 的误差反馈完全失效,SNR 看着仍有 9 dB 量级。
                Hm = torch.eye(in_n)
                degraded = True
            else:
                Hm = H.float() / max(int(ntok[e]), 1) if ntok is not None else H.float()
            Hm = core.damp_hessian(Hm, damp)
            Wh = core.unisvq_qdq(
                Wf,
                Hm,
                core.rand_signs(in_n, sign_seed(layer, e, f"{tag}_in")),
                core.rand_signs(out_n, sign_seed(layer, e, f"{tag}_out")),
                Hd,
                cb,
                use_ldlq=use_ldlq,
            )
            rec[f"{tag}_snr"] = round(core.snr_db(Wf, Wh), 4)
            rec[f"{tag}_proxy"] = core.proxy_loss(Wf, Wh, Hm)
            if degraded:
                rec[f"{tag}_degraded"] = True
            (out13 if tag == "w13" else out2)[e] = Wh.to(W.dtype)
        if ntok is not None:
            rec["ntok"] = int(ntok[e])
        recs.append(rec)
    return out13, out2, recs


# ---------------------------------------------------------------- 驱动
def parse_layers(spec: str | None, available: list[int]) -> list[int]:
    if not spec or spec == "all":
        return sorted(available)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    missing = [ln for ln in out if ln not in available]
    if missing:
        die(f"checkpoint 里没有这些层:{missing}", f"可用:{sorted(available)[:8]}...")
    return sorted(set(out))


def cmd_run(args) -> int:
    src = args.src
    idx = index_checkpoint(src)
    geo = read_geometry(src)
    layers = parse_layers(args.layers, list(idx))

    probe = idx[layers[0]]
    gu_shape, dn_shape = probe["gate_up"][2], probe["down"][2]
    e_n = gu_shape[0]
    if geo["num_experts"] not in (None, e_n):
        die(f"config.json 说 num_experts={geo['num_experts']},张量第 0 维是 {e_n}")
    # 两种朝向下 (dim, hidden) 的候选:gate_up 的两维一个是 dim 一个是 2*hidden
    a, b = gu_shape[1], gu_shape[2]
    cands = [(b, a // 2), (a, b // 2)]
    dim = hidden = None
    for d_c, h_c in cands:
        if sorted(dn_shape[1:]) == sorted([d_c, h_c]):
            dim, hidden = d_c, h_c
            break
    if dim is None:
        die(f"gate_up {gu_shape} 和 down {dn_shape} 推不出一致的 (dim, hidden)")
    layout = decide_layout(gu_shape, dn_shape, dim, hidden, args.layout)

    top_k = args.top_k or geo["top_k"]
    use_hessian = args.hessian != "identity"
    if use_hessian and not args.acts:
        die("--hessian 不是 identity 时必须给 --acts", "或者显式用 --hessian identity(H=I,没有校准的 QuIP)。")
    if use_hessian and not top_k:
        die("推不出 top_k", "config.json 里没有 num_experts_per_tok,用 --top-k 给。")

    if os.path.exists(args.out) and os.listdir(args.out):
        die(f"输出目录非空: {args.out}(拒绝覆盖)")
    os.makedirs(args.out, exist_ok=True)

    keep = load_keep_hot(args.keep_hot, e_n, layers) if args.keep_hot else None
    bitacct = avg_bits_report(args.bits, keep, layers, e_n)

    print("================ 配置 ================")
    print(f"E / dim / hidden   {e_n} / {dim} / {hidden}")
    print(f"朝向               {layout}" + ("  (按形状判出)" if args.layout == "auto" else "  (显式指定)"))
    print(f"gate_up 顺序       {'[gate; up]' if args.gate_up_order == 'gate_first' else '[up; gate]'}")
    print(f"位宽 / 码本维度    {args.bits} bit / d={core.D_GROUP}")
    print(f"Hadamard 块        {core.HAD_BLOCK}")
    print(f"Hessian            {'来自 ' + args.acts if use_hessian else 'identity(无校准)'}")
    print(f"阻尼               {args.damp}")
    print(f"LDLQ               {'开' if not args.no_ldlq else '关(退化成逐组码本量化)'}")
    print(f"层                 {len(layers)} 层: {layers if len(layers) <= 8 else str(layers[:8]) + '...'}")
    if keep:
        print(f"混合精度回退       {args.keep_hot}")
        print(f"                   保留 {bitacct['kept_experts']}/{bitacct['expert_slots']} 个专家槽"
              f"({100 * bitacct['kept_ratio']:.2f}%),保留的是 **bf16**")
        print(f"平均位宽           {bitacct['avg_bits']:.3f} bit"
              f"  = {args.bits} + (16-{args.bits})x{bitacct['kept_ratio']:.4f}")
        print(f"                   同保留比例换成 int8 容器会是 "
              f"{args.bits + (8 - args.bits) * bitacct['kept_ratio']:.3f} bit —— "
              f"DSv4.1 的数是后者那个口径,跨路线比较前先对齐(见 mp_select.py 的位宽账)")
    else:
        print(f"混合精度回退       无(全部专家都量化到 {args.bits} bit)")
    if args.gate_up_order == "gate_first" and use_hessian:
        print(
            "\n注意:gate/up 顺序弄反会让 H2 整个错掉,而 SNR 看不出来"
            "\n      (SNR 只看 ||W-Ŵ||,不看 H)。顺序已记进 manifest。"
        )

    manifest = {
        "source": os.path.abspath(src),
        "method": "unisvq_rht_ldlq" if not args.no_ldlq else "unisvq_rht_codebook",
        "bits": args.bits,
        "d_group": core.D_GROUP,
        "had_block": core.HAD_BLOCK,
        "layout": layout,
        "gate_up_order": args.gate_up_order,
        "hessian": "identity" if not use_hessian else os.path.abspath(args.acts),
        "damp": args.damp,
        "top_k": top_k if use_hessian else None,
        "num_experts": e_n,
        "dim": dim,
        "moe_hidden_dim": hidden,
        "sign_seed_scheme": SIGN_SEED_SCHEME,
        "scope": "routed experts only",
        "keep_hot": os.path.abspath(args.keep_hot) if args.keep_hot else None,
        "mixed_precision": bitacct,
        "layers": {},
    }

    # 量化结果按分片归集,最后一次性写出(一个分片可能含多层)
    patched: dict[str, dict[str, torch.Tensor]] = {}
    t0 = time.time()
    for ln in layers:
        d = idx[ln]
        w13 = as_out_in(get_tensor(src, d["gate_up"][0], d["gate_up"][1]), layout)
        w2 = as_out_in(get_tensor(src, d["down"][0], d["down"][1]), layout)
        H13 = H2 = ntok = None
        if use_hessian:
            if "router" not in d:
                die(f"第 {ln} 层没有 router 权重", "H13/H2 要靠 router 分 token;--hessian identity 可以跳过。")
            router_w = get_tensor(src, d["router"][0], d["router"][1])
            x = load_acts(args.acts, ln, dim)
            H13, H2, ntok = expert_hessians(
                x, router_w, w13, top_k, args.gate_up_order == "gate_first", args.max_tok
            )
            zero = int((ntok == 0).sum())
            print(f"\n[层 {ln}] 激活 {x.shape[0]} token;专家命中 ntok 中位 {int(ntok.median())},零命中 {zero} 个")
            if zero:
                print(f"         !! {zero} 个专家没有 token —— 这些矩阵会退化成 H=I 并记成 DEGRADED")
        else:
            print(f"\n[层 {ln}] H=identity")

        keep_ln = keep.get(ln) if keep else None
        q13, q2, recs = quantize_layer(
            w13, w2, H13, H2, ntok, ln, args.bits, args.damp, not args.no_ldlq, keep_ln
        )
        patched.setdefault(d["gate_up"][0], {})[d["gate_up"][1]] = from_out_in(q13, layout)
        patched.setdefault(d["down"][0], {})[d["down"][1]] = from_out_in(q2, layout)

        # 保留的专家没有 SNR(未量化,+inf),统计只看真被量化的那些。
        s13 = sorted(r["w13_snr"] for r in recs if "w13_snr" in r)
        s2 = sorted(r["w2_snr"] for r in recs if "w2_snr" in r)
        n_kept = sum(1 for r in recs if r.get("kept"))
        if not s13:
            die(f"第 {ln} 层一个专家都没量化", "keep 列表覆盖了整层?")
        deg = sum(1 for r in recs if r.get("w13_degraded") or r.get("w2_degraded"))
        print(
            f"         w13 SNR min {s13[0]:.2f} / p50 {s13[len(s13) // 2]:.2f} / max {s13[-1]:.2f} dB"
            f"   w2 SNR min {s2[0]:.2f} / p50 {s2[len(s2) // 2]:.2f} / max {s2[-1]:.2f} dB"
            + (f"   保留 {n_kept}" if n_kept else "")
            + (f"   DEGRADED {deg}" if deg else "")
        )
        if n_kept and ntok is not None:
            kept_tok = sum(int(ntok[e]) for e in keep_ln)
            tot_tok = max(int(ntok.sum()), 1)
            print(f"         保留的 {n_kept} 个专家承载本层校准 token 的 "
                  f"{100.0 * kept_tok / tot_tok:.1f}%")
        manifest["layers"][str(ln)] = {
            "w13_snr": {"min": s13[0], "p50": s13[len(s13) // 2], "max": s13[-1]},
            "w2_snr": {"min": s2[0], "p50": s2[len(s2) // 2], "max": s2[-1]},
            "degraded": deg,
            "kept": sorted(keep_ln) if keep_ln else [],
            "experts": recs if args.full_manifest else None,
        }

    # ---- 写出。只有被量化的层所在分片会重写,其余原样复制。
    print("\n================ 写出 ================")
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".safetensors")):
        sp, dp = os.path.join(src, fn), os.path.join(args.out, fn)
        if fn not in patched:
            shutil.copy2(sp, dp)
            print(f"  [copy]  {fn}")
            continue
        tensors: dict[str, torch.Tensor] = {}
        meta: dict[str, str] = {}
        with safe_open(sp, framework="pt") as f:
            meta.update(f.metadata() or {})
            for k in f.keys():  # noqa: SIM118
                # 不能写成 `patched[fn].get(k) or f.get_tensor(k)` ——
                # `or` 会对张量取 __bool__,多元素张量直接抛。
                q = patched[fn].get(k)
                tensors[k] = f.get_tensor(k) if q is None else q
        meta["ptq_method"] = manifest["method"]
        meta["ptq_bits"] = str(args.bits)
        meta["ptq_layout"] = layout
        meta["ptq_gate_up_order"] = args.gate_up_order
        # 平均位宽进 metadata:评测端拿到分片就能知道这是几 bit 的产物,
        # 不必回头去翻 manifest。TAG 也该照这个数起(工单 §4.2)。
        meta["ptq_avg_bits"] = str(bitacct["avg_bits"])
        meta["ptq_kept_experts"] = str(bitacct["kept_experts"])
        meta.setdefault("format", "pt")
        save_file(tensors, dp, metadata=meta)
        print(f"  [quant] {fn}  ({len(patched[fn])} 个张量被替换)")

    for fn in sorted(os.listdir(src)):
        if fn.endswith(".safetensors") or os.path.isdir(os.path.join(src, fn)):
            continue
        shutil.copy2(os.path.join(src, fn), os.path.join(args.out, fn))

    dt = time.time() - t0
    manifest["seconds"] = round(dt, 1)
    manifest["quantized_layers"] = layers
    with open(os.path.join(args.out, "ptq_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    all13 = [v["w13_snr"]["p50"] for v in manifest["layers"].values()]
    all2 = [v["w2_snr"]["p50"] for v in manifest["layers"].values()]
    tot_deg = sum(v["degraded"] for v in manifest["layers"].values())
    print(f"\n{len(layers)} 层,{dt / 60:.1f} 分钟")
    print(f"w13 p50 SNR  {min(all13):.2f} .. {max(all13):.2f} dB")
    print(f"w2  p50 SNR  {min(all2):.2f} .. {max(all2):.2f} dB")
    print(f"DEGRADED     {tot_deg} 个矩阵" + ("  <- 工单 §2.4 要求全量时为 0" if tot_deg else ""))
    if keep:
        print(f"平均位宽     {bitacct['avg_bits']:.3f} bit"
              f"(保留 {bitacct['kept_experts']}/{bitacct['expert_slots']} 个专家槽为 bf16)")
    print(f"manifest ->  {os.path.join(args.out, 'ptq_manifest.json')}")
    if len(layers) < len(idx):
        print(f"\n这是 {len(layers)}/{len(idx)} 层的部分产物 —— 其余层仍是原权重,**不能用来出交付数**。")
    return 1 if (tot_deg and len(layers) == len(idx)) else 0


def cmd_hessian(args) -> int:
    """只算 Hessian 存盘。给调参/复用,不是主路径。"""
    idx = index_checkpoint(args.src)
    geo = read_geometry(args.src)
    layers = parse_layers(args.layers, list(idx))
    probe = idx[layers[0]]
    gu_shape, dn_shape = probe["gate_up"][2], probe["down"][2]
    a, b = gu_shape[1], gu_shape[2]
    dim = hidden = None
    for d_c, h_c in [(b, a // 2), (a, b // 2)]:
        if sorted(dn_shape[1:]) == sorted([d_c, h_c]):
            dim, hidden = d_c, h_c
            break
    if dim is None:
        die("推不出 (dim, hidden)")
    layout = decide_layout(gu_shape, dn_shape, dim, hidden, args.layout)
    top_k = args.top_k or geo["top_k"]
    if not top_k:
        die("推不出 top_k,用 --top-k 给")
    os.makedirs(args.out, exist_ok=True)
    for ln in layers:
        d = idx[ln]
        if "router" not in d:
            die(f"第 {ln} 层没有 router 权重")
        w13 = as_out_in(get_tensor(args.src, d["gate_up"][0], d["gate_up"][1]), layout)
        x = load_acts(args.acts, ln, dim)
        H13, H2, ntok = expert_hessians(
            x, get_tensor(args.src, d["router"][0], d["router"][1]), w13,
            top_k, args.gate_up_order == "gate_first", args.max_tok,
        )
        path = os.path.join(args.out, f"hessian_layer{ln:02d}.pt")
        torch.save({"H13": H13, "H2": H2, "ntok": ntok, "top_k": top_k,
                    "gate_up_order": args.gate_up_order, "layout": layout}, path)
        print(f"[层 {ln}] ntok 中位 {int(ntok.median())},零命中 {int((ntok == 0).sum())}  -> {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--in", dest="src", required=True, help="源 HF checkpoint 目录")
        p.add_argument("--acts", help="各层 MoE 输入激活目录,每层一个 layer<NN>.pt")
        p.add_argument("--layers", help="要处理的层,如 20 / 0-3 / 0,5,9 / all(默认 all)")
        p.add_argument("--layout", choices=("auto", "efd", "edf"), default="auto")
        p.add_argument(
            "--gate-up-order",
            choices=("gate_first", "up_first"),
            default="gate_first",
            help="融合的是 [gate;up] 还是 [up;gate]。**弄反会让 H2 静默错掉**,见 docstring。",
        )
        p.add_argument("--top-k", type=int, help="router top-k;默认从 config.json 读")
        p.add_argument("--max-tok", type=int, default=0, help=">0 时每个专家最多采这么多 token 累 Hessian")

    pr = sub.add_parser("run", help="算 Hessian + 量化 + 写 checkpoint")
    common(pr)
    pr.add_argument("--out", required=True, help="输出 checkpoint 目录")
    pr.add_argument("--bits", type=int, required=True)
    pr.add_argument("--hessian", default="acts", help="'identity' 表示 H=I(无校准);否则用 --acts")
    pr.add_argument("--damp", type=float, default=0.01, help="Tikhonov 阻尼系数(相对 mean diag)")
    pr.add_argument("--no-ldlq", action="store_true", help="关掉 LDLQ,只留 RHT + 码本")
    pr.add_argument(
        "--keep-hot",
        help="混合精度回退(工单 §4.2):mp_select.py 产出的 keep json,"
             "列进来的专家不量化、保持 bf16。平均位宽记进 manifest 和分片 metadata。",
    )
    pr.add_argument("--full-manifest", action="store_true", help="manifest 里记每个专家的 SNR/proxy")
    pr.set_defaults(func=cmd_run)

    ph = sub.add_parser("hessian", help="只算 Hessian 存盘")
    common(ph)
    ph.add_argument("--out", required=True, help="Hessian 输出目录")
    ph.set_defaults(func=cmd_hessian)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
