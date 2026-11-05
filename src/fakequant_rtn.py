#!/usr/bin/env python3
"""工单 1.1:把 HF checkpoint 的**路由专家**权重做 N-bit 假量化,写出一份 bf16 checkpoint。

这是什么
--------
量化再反量化(qdq),结果仍按原 dtype 存。所以产出的 checkpoint 能被任何
普通 bf16 推理/训练路径直接加载,**不需要低位宽 kernel**。
位宽是 Ŵ 的信息量属性,不是容器属性 —— 容器还是 bf16。

用 RTN(逐组对称舍入,不需要 Hessian、不需要校准)。实测在
Qwen3.6-35B-A3B 的真实形状上:

    bits   RTN(group128)   LDLQ+RHT
      2        2.31 dB      8.67 dB     <- 差 6.4 dB,2bit 别用 RTN
      3       11.26 dB     12.04 dB     <- 差 0.8 dB,够当脚手架

所以 **RTN 只用于 3bit 及以上的管道打通**;2bit 要等 `offline_ptq.py`。

范围:只做路由专家
------------------
和 DSv4.1 一致。**共享专家、attention、router gate、norm、embedding、lm_head
一律不碰。** 范围判定不靠假设:先 `--scan` 把分类打出来人工核对,
再 `--apply`。任何"看着像权重但没被分类"的键都会让脚本**报错退出**,
不会静默跳过 —— 静默跳过会产出一个看着正常、实际只量化了一部分的 checkpoint。

用法
----
    # 1) 先看分类对不对(不写任何文件)
    python3 fakequant_rtn.py --in <hf_ckpt> --scan

    # 2) 确认后再做
    python3 fakequant_rtn.py --in <hf_ckpt> --out <dst> --bits 3 --group 128

    # 3) 独立校验(重新读回产物逐张量比对)
    python3 fakequant_rtn.py --in <hf_ckpt> --out <dst> --bits 3 --verify-only

只依赖 torch + safetensors。不需要 NPU、不需要 torchtitan。
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
try:
    from unisvq_core import rtn_group_qdq, snr_db
except ImportError:  # 允许 src/ 不在 sys.path 时从同级目录找
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from unisvq_core import rtn_group_qdq, snr_db

# ---------------------------------------------------------------- 键名分类
# 路由专家。两种已知布局都覆盖:
#   逐专家:  model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
#   融合:    model.layers.{i}.mlp.experts.{gate_up,down}_proj[.weight]   (3D: [E, ...])
#
# Qwen3.6-35B-A3B 实测(2026-11-05,官方权重)走的是**融合布局**,而且外面套了
# 一层多模态前缀:
#   model.language_model.layers.{i}.mlp.experts.gate_up_proj   [256, 1024, 2048]
#   model.language_model.layers.{i}.mlp.experts.down_proj      [256, 2048, 512]
# 共 80 个张量(40 层 x 2)、32.212 G 参数,和解析式
#   40 x 256 x (1024x2048 + 2048x512) = 32.212 G
# 对得上。1024 = 2 x 512 是 gate 与 up 融合在输出维。
#
# **融合顺序([gate;up] 还是 [up;gate])对本脚本不影响**:qdq 是按输出行独立做的,
# 每行自己一套 scale,gate 行和 up 行不会互相污染。顺序只在 LoRA 合并时才要紧
# (工单 §4.5 的风险项),到那一步再用 verify_recon.py 定。
ROUTED_PATTERNS = (
    r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$",
    r"\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)(\.weight)?$",
    r"\.routed_experts\.inner_experts\.(w1_EFD|w2_EDF|w3_EFD)$",
)
# 明确排除。**顺序重要**:排除优先于命中,`shared_expert` 的键里也含 `_proj`。
EXCLUDE_PATTERNS = (
    r"shared_expert",          # 共享专家:原精度
    r"\.mlp\.gate\.",          # router 的 gate
    r"\.mlp\.router\.",
    # 注意力,**所有变体**。写成 `attn\.` 而不是枚举 `self_attn`,是因为
    # Qwen3.6-35B-A3B 实测是混合架构:40 层里 30 层是 Gated Delta Net
    # (`layers.N.linear_attn.*`,带 A_log / dt_bias / conv1d 这些 SSM 参数),
    # 只有 10 层是全注意力。2026-11-05 第一次对真权重扫描时,`self_attn|attention`
    # 两个都匹配不到 `linear_attn`,240 个张量落进了 unknown —— 守卫因此拦住了,
    # 这正是它该做的。
    # 为什么线性注意力也排除:它是注意力不是专家(范围见工单 §1.5);
    # 只有 1.012 G 参数(2.9%),量到 2bit 省不到 1 GB;而且 conv1d.weight 是
    # [8192, 1, 4],每通道只有 4 个元素,group=128 除不开,分组量化没有意义。
    r"attn\.|attention|_attn$",
    r"embed_tokens|tok_embeddings",
    r"lm_head",
    r"norm",                   # RMSNorm / LayerNorm
    r"\.bias$",
    r"rotary|inv_freq",
    r"mtp|vision|visual|patch_embed",  # MTP 与 ViT:本阶段不量化
    r"_scale|_zero_point|scale_inv",   # 已有的量化元数据
)
# 已知可以安全忽略的非权重键(不算"未分类")
BENIGN_PATTERNS = (r"\.bias$", r"norm", r"inv_freq", r"_scale", r"_zero_point")

_ROUTED_RE = [re.compile(p) for p in ROUTED_PATTERNS]
_EXCL_RE = [re.compile(p) for p in EXCLUDE_PATTERNS]
_BENIGN_RE = [re.compile(p) for p in BENIGN_PATTERNS]


def classify(key: str) -> str:
    """返回 'routed' / 'excluded' / 'unknown'。排除优先。"""
    for r in _EXCL_RE:
        if r.search(key):
            return "excluded"
    for r in _ROUTED_RE:
        if r.search(key):
            return "routed"
    for r in _BENIGN_RE:
        if r.search(key):
            return "excluded"
    return "unknown"


# ---------------------------------------------------------------- checkpoint IO
def list_shards(src: str) -> list[str]:
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"!! {src} 下没有 .safetensors")
    return files


def scan(src: str) -> dict:
    """只读 header,不载张量。"""
    buckets: dict[str, list[tuple[str, str, list[int]]]] = {"routed": [], "excluded": [], "unknown": []}
    per_shard: dict[str, int] = {}
    for fn in list_shards(src):
        path = os.path.join(src, fn)
        with safe_open(path, framework="pt") as f:
            n_routed = 0
            for k in f.keys():  # noqa: SIM118
                sl = f.get_slice(k)
                kind = classify(k)
                buckets[kind].append((k, sl.get_dtype(), list(sl.get_shape())))
                if kind == "routed":
                    n_routed += 1
        per_shard[fn] = n_routed
    return {"buckets": buckets, "per_shard": per_shard}


def numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def report_scan(info: dict) -> int:
    b = info["buckets"]
    print("\n================ 键名分类 ================")
    for kind in ("routed", "excluded", "unknown"):
        items = b[kind]
        tot = sum(numel(s) for _, _, s in items)
        print(f"\n[{kind}]  {len(items)} 个张量,{tot / 1e9:.3f} G 参数")
        for k, dt, s in items[:4]:
            print(f"    {k}\n        dtype={dt} shape={s}")
        if len(items) > 4:
            print(f"    ... 其余 {len(items) - 4} 个")

    n_shard_with = sum(1 for v in info["per_shard"].values() if v)
    print(f"\n含路由专家的分片: {n_shard_with} / {len(info['per_shard'])}")

    if b["unknown"]:
        print("\n!! 有未分类的键 —— 拒绝继续。")
        print("   要么它们该被量化(加进 ROUTED_PATTERNS),要么该被排除")
        print("   (加进 EXCLUDE_PATTERNS)。**不要让脚本静默跳过。**")
        for k, _, s in b["unknown"][:20]:
            print(f"     {k}  {s}")
        return 1
    if not b["routed"]:
        print("\n!! 没有识别出任何路由专家 —— 布局和预期不符,先核对键名")
        return 1
    print("\n分类无歧义。")
    return 0


# ---------------------------------------------------------------- 量化
@torch.no_grad()
def quant_tensor(t: torch.Tensor, bits: int, group: int) -> tuple[torch.Tensor, float]:
    """对 2D [out, in] 或 3D [E, out, in] 做逐组 RTN qdq。返回 (Ŵ, SNR)。"""
    orig_dtype = t.dtype
    w = t.float()
    if w.ndim == 2:
        wh = rtn_group_qdq(w, bits=bits, group=group)
    elif w.ndim == 3:
        wh = torch.stack([rtn_group_qdq(w[e], bits=bits, group=group) for e in range(w.shape[0])])
    else:
        raise ValueError(f"只支持 2D/3D,得到 {tuple(t.shape)}")
    return wh.to(orig_dtype), snr_db(w, wh.float())


def apply_quant(src: str, dst: str, bits: int, group: int) -> dict:
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"!! 输出目录非空: {dst}(拒绝覆盖)")
    os.makedirs(dst, exist_ok=True)

    info = scan(src)
    if report_scan(info) != 0:
        raise SystemExit("!! 扫描未通过,不动手")

    manifest = {
        "source": os.path.abspath(src),
        "bits": bits,
        "group": group,
        "method": "rtn_group_symmetric_qdq",
        "container_dtype": "原 dtype 不变(通常 bf16)",
        "scope": "routed experts only",
        "tensors": {},
    }
    t_start = time.time()
    n_q = 0
    for fn in list_shards(src):
        sp, dp = os.path.join(src, fn), os.path.join(dst, fn)
        if info["per_shard"][fn] == 0:
            # 本分片没有量化目标 —— 原样复制,不重写(省时间也避免改动无关张量)
            shutil.copy2(sp, dp)
            print(f"  [copy] {fn}")
            continue
        tensors: dict[str, torch.Tensor] = {}
        meta: dict[str, str] = {}
        with safe_open(sp, framework="pt") as f:
            md = f.metadata()
            if md:
                meta.update(md)
            for k in f.keys():  # noqa: SIM118
                t = f.get_tensor(k)
                if classify(k) == "routed":
                    t, snr = quant_tensor(t, bits, group)
                    manifest["tensors"][k] = round(snr, 4)
                    n_q += 1
                tensors[k] = t
        meta["fakequant_bits"] = str(bits)
        meta["fakequant_group"] = str(group)
        meta["fakequant_scope"] = "routed_experts_only"
        save_file(tensors, dp, metadata=meta)
        snrs = [manifest["tensors"][k] for k in tensors if k in manifest["tensors"]]
        print(f"  [quant] {fn}  {len(snrs)} 个张量  SNR 中位 {sorted(snrs)[len(snrs) // 2]:.2f} dB")

    # 非权重文件原样带过去(config.json / tokenizer / index.json ...)
    for fn in sorted(os.listdir(src)):
        if fn.endswith(".safetensors") or os.path.isdir(os.path.join(src, fn)):
            continue
        shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
        print(f"  [copy] {fn}")

    dt = time.time() - t_start
    all_snr = sorted(manifest["tensors"].values())
    manifest["n_quantized"] = n_q
    manifest["snr_db"] = {
        "min": all_snr[0],
        "p50": all_snr[len(all_snr) // 2],
        "max": all_snr[-1],
    }
    manifest["seconds"] = round(dt, 1)
    with open(os.path.join(dst, "fakequant_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"\n量化了 {n_q} 个张量,{dt / 60:.1f} 分钟")
    print(f"SNR  min {all_snr[0]:.2f} / p50 {manifest['snr_db']['p50']:.2f} / max {all_snr[-1]:.2f} dB")
    print(f"manifest -> {os.path.join(dst, 'fakequant_manifest.json')}")
    return manifest


# ---------------------------------------------------------------- 校验
@torch.no_grad()
def verify(src: str, dst: str, bits: int, group: int, limit: int = 8) -> int:
    """独立重算:抽查若干路由专家张量,产物必须与现场重算逐位一致;
    同时确认被排除的张量**没有被改动**。"""
    print("\n================ 独立校验 ================")
    bad = 0
    checked_q = checked_x = 0
    for fn in list_shards(src):
        sp, dp = os.path.join(src, fn), os.path.join(dst, fn)
        if not os.path.exists(dp):
            print(f"  X  产物缺少分片 {fn}")
            bad += 1
            continue
        with safe_open(sp, framework="pt") as fs, safe_open(dp, framework="pt") as fd:
            for k in fs.keys():  # noqa: SIM118
                kind = classify(k)
                if kind == "routed" and checked_q < limit:
                    ref, _ = quant_tensor(fs.get_tensor(k), bits, group)
                    got = fd.get_tensor(k)
                    same = torch.equal(ref, got)
                    print(f"  {'OK ' if same else 'X  '} 量化一致 {k}")
                    bad += 0 if same else 1
                    checked_q += 1
                elif kind == "excluded" and checked_x < limit:
                    same = torch.equal(fs.get_tensor(k), fd.get_tensor(k))
                    print(f"  {'OK ' if same else 'X  '} 未被改动 {k}")
                    bad += 0 if same else 1
                    checked_x += 1
        if checked_q >= limit and checked_x >= limit:
            break
    print("\n校验" + ("通过。" if bad == 0 else f"失败 {bad} 项。"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="源 HF checkpoint 目录")
    ap.add_argument("--out", dest="dst", help="输出目录(必须不存在或为空)")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128, help="沿输入维的分组大小")
    ap.add_argument("--scan", action="store_true", help="只扫描键名分类,不写文件")
    ap.add_argument("--verify-only", action="store_true", help="只对已有产物做独立校验")
    a = ap.parse_args()

    if a.scan:
        return report_scan(scan(a.src))
    if not a.dst:
        return ap.error("--out 是必需的(除了 --scan)")
    if a.bits < 2:
        return ap.error("bits 至少 2")
    if a.bits == 2:
        print("!! 警告:2bit 下 RTN 比 LDLQ+RHT 差约 6.4 dB(实测 2.31 vs 8.67)。")
        print("   RTN 只适合 3bit 及以上的管道打通;2bit 请用 offline_ptq.py。")
    if a.verify_only:
        return verify(a.src, a.dst, a.bits, a.group)
    apply_quant(a.src, a.dst, a.bits, a.group)
    return verify(a.src, a.dst, a.bits, a.group)


if __name__ == "__main__":
    sys.exit(main())
