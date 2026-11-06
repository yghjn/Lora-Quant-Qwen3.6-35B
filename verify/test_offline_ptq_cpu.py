#!/usr/bin/env python
"""offline_ptq.py 的自测,纯 CPU、不占卡、不需要真权重和真激活。

    python verify/test_offline_ptq_cpu.py

分三段:

**A. Hessian 的数学**(不经过子进程,直接调函数)
  1. `route` 的 topk 和独立算的 logits 排序一致
  2. `H13[e]` 逐元素等于该专家命中 token 的 `xe^T xe`
  3. `H2[e]` 等于独立算的 `a^T a`,`a = silu(gate) * up`
  4. **gate/up 顺序换一下,H2 必须变**(这是会静默出错的那处)
  5. 没被任何 token 命中的专家 `ntok == 0`,两个 H 全零
  6. `sign_seed` 对 (layer, expert, tag) 不碰撞,且可复现

**B. 端到端**(走子进程,两种朝向各一次)
  7. 退出码 0,产物键名齐全,非专家张量逐位未变
  8. SNR 在合理区间,`proxy_loss` 比 RTN 低
  9. manifest 记了朝向、顺序、种子方案
 10. `--hessian identity` 不给 --acts 也能跑
 11. `ntok=0` 的专家被记成 DEGRADED
 12. 输出目录非空时拒绝;朝向撞车时拒绝

**C. 一个会影响排期的测量**
 13. 2bit 下三档对比:RTN / RHT+码本(无 LDLQ)/ RHT+LDLQ,H=I 与真 H

几何取 `dim=384 / hidden=128`。两个约束同时要满足:

* `dim` 和 `hidden` 都要被 `HAD_BLOCK=128` 整除(w2 的输入维就是 hidden);
* `dim != 2*hidden`,否则两种朝向形状一样、判不出来。

所以 `hidden=128` 时 `dim` 不能取 256。真模型是 2048 / 512,同样满足。
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
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import offline_ptq as PTQ  # noqa: E402
import unisvq_core as core  # noqa: E402

SCRIPT = os.path.join(HERE, "..", "src", "offline_ptq.py")
E, D, F, TOPK = 4, 384, 128, 2  # D != 2F,且两者都被 HAD_BLOCK=128 整除
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


def run(*extra: str) -> subprocess.CompletedProcess:
    argv = [sys.executable] + (["-I"] if sys.flags.isolated else []) + [SCRIPT, *extra]
    return subprocess.run(argv, capture_output=True, text=True)


# ---------------------------------------------------------------- 夹具
def make_ckpt(dirpath: str, layout: str) -> dict:
    """合成一个 HF 风格 checkpoint。

    **不要试图靠改 router 权重造"死专家"。** logits 是 `x @ w.T`,没有 bias,
    所以把某一行设成大负数只会让 `sum(x) < 0` 的 token 拿到巨大**正**值,
    那个专家反而更容易入选 —— 第一版夹具就是这么错的(实测 ntok=259 而非 0)。
    造死专家的确定做法是让激活每一行都相同:路由对所有 token 一致,
    于是恰好 `E - top_k` 个专家一个 token 都拿不到。见 `make_const_acts`。
    """
    g = torch.Generator().manual_seed(5)
    t: dict[str, torch.Tensor] = {}
    for layer in LAYERS:
        p = f"model.language_model.layers.{layer}.mlp."
        if layout == "efd":
            t[p + "experts.gate_up_proj"] = (torch.randn(E, 2 * F, D, generator=g) * 0.05).bfloat16()
            t[p + "experts.down_proj"] = (torch.randn(E, D, F, generator=g) * 0.05).bfloat16()
        else:
            t[p + "experts.gate_up_proj"] = (torch.randn(E, D, 2 * F, generator=g) * 0.05).bfloat16()
            t[p + "experts.down_proj"] = (torch.randn(E, F, D, generator=g) * 0.05).bfloat16()
        t[p + "gate.weight"] = torch.randn(E, D, generator=g).bfloat16()
        # 不该被动的键
        t[p + "shared_expert.gate_proj.weight"] = torch.randn(F, D, generator=g).bfloat16()
    os.makedirs(dirpath, exist_ok=True)
    save_file(t, os.path.join(dirpath, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(dirpath, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "qwen3_5_moe", "num_experts": E, "num_experts_per_tok": TOPK}, fh)
    return t


def make_acts(dirpath: str, n: int = 512) -> None:
    os.makedirs(dirpath, exist_ok=True)
    g = torch.Generator().manual_seed(9)
    for layer in LAYERS:
        torch.save(torch.randn(n, D, generator=g).bfloat16(), os.path.join(dirpath, f"layer{layer}.pt"))


def make_const_acts(dirpath: str, n: int = 256) -> None:
    """每一行都相同的激活 -> 路由对所有 token 一致 -> 恰好 E-top_k 个专家 ntok=0。"""
    os.makedirs(dirpath, exist_ok=True)
    g = torch.Generator().manual_seed(21)
    row = torch.randn(1, D, generator=g)
    for layer in LAYERS:
        torch.save(row.expand(n, D).contiguous().bfloat16(), os.path.join(dirpath, f"layer{layer}.pt"))


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


# ---------------------------------------------------------------- A
def section_a() -> None:
    print("\n=== A. Hessian 的数学 ===")
    g = torch.Generator().manual_seed(3)
    x = torch.randn(300, D, generator=g)
    router_w = torch.randn(E, D, generator=g)
    w13 = torch.randn(E, 2 * F, D, generator=g) * 0.05

    ids = PTQ.route(x, router_w, TOPK)
    ref_logits = x.float() @ router_w.float().T
    ref_ids = torch.topk(ref_logits, k=TOPK, dim=-1).indices
    check("route 的 topk 与独立算的一致", torch.equal(ids, ref_ids))

    H13, H2, ntok = PTQ.expert_hessians(x, router_w, w13, TOPK, gate_first=True)

    worst13 = worst2 = 0.0
    for e in range(E):
        sel = (ids == e).any(dim=-1).nonzero(as_tuple=True)[0]
        check(f"专家 {e} 的 ntok 对得上", int(ntok[e]) == sel.numel(), f"{int(ntok[e])} vs {sel.numel()}")
        if sel.numel() == 0:
            continue
        xe = x.float()[sel]
        ref = xe.T @ xe
        worst13 = max(worst13, (H13[e] - ref).abs().max().item() / ref.abs().max().item())
        gu = xe @ w13[e].float().T
        a = torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]
        ref2 = a.T @ a
        worst2 = max(worst2, (H2[e] - ref2).abs().max().item() / ref2.abs().max().item())
    check(f"H13 == xe^T xe (相对误差 {worst13:.2e})", worst13 < 1e-5)
    check(f"H2 == a^T a   (相对误差 {worst2:.2e})", worst2 < 1e-5)

    _, H2_swapped, _ = PTQ.expert_hessians(x, router_w, w13, TOPK, gate_first=False)
    diff = (H2 - H2_swapped).abs().max().item() / H2.abs().max().clamp_min(1e-9).item()
    check(f"gate/up 顺序互换后 H2 确实变了 (相对差 {diff:.2f})", diff > 0.1)

    # sign_seed
    seeds = {
        (ln, e, tag): PTQ.sign_seed(ln, e, tag)
        for ln in range(40)
        for e in range(256)
        for tag in ("w13_in", "w13_out", "w2_in", "w2_out")
    }
    check(f"sign_seed 在 40x256x4 上无碰撞 ({len(set(seeds.values()))} 个不同值)",
          len(set(seeds.values())) == len(seeds))
    check("sign_seed 可复现", PTQ.sign_seed(7, 33, "w2_in") == PTQ.sign_seed(7, 33, "w2_in"))
    s1 = core.rand_signs(D, PTQ.sign_seed(7, 33, "w2_in"))
    s2 = core.rand_signs(D, PTQ.sign_seed(7, 33, "w2_in"))
    check("同种子的 rand_signs 逐位相同", torch.equal(s1, s2))


# ---------------------------------------------------------------- B
def section_b(tmp: str, layout: str) -> None:
    print(f"\n=== B. 端到端 (朝向 {layout}) ===")
    src = os.path.join(tmp, f"src_{layout}")
    dst = os.path.join(tmp, f"dst_{layout}")
    acts = os.path.join(tmp, "acts")
    base = make_ckpt(src, layout)
    p = run("run", "--in", src, "--out", dst, "--bits", "3", "--acts", acts,
            "--layers", "0", "--full-manifest")
    ok = p.returncode == 0
    check(f"{layout}: 退出码 0", ok, p.stdout[-1500:] + p.stderr[-1000:])
    if not ok:
        return

    got = read_dir(dst)
    check(f"{layout}: 键集合不变", set(got) == set(base))

    p0 = "model.language_model.layers.0.mlp."
    p1 = "model.language_model.layers.1.mlp."
    for key, changed in ((p0 + "experts.gate_up_proj", True), (p0 + "experts.down_proj", True),
                         (p1 + "experts.gate_up_proj", False),
                         (p0 + "gate.weight", False),
                         (p0 + "shared_expert.gate_proj.weight", False)):
        same = torch.equal(got[key], base[key])
        name = key.replace("model.language_model.", "")
        check(f"{layout}: {name} {'被量化' if changed else '未改动'}", same != changed)

    with open(os.path.join(dst, "ptq_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    check(f"{layout}: manifest 记了朝向 {layout}", man.get("layout") == layout, str(man.get("layout")))
    check(f"{layout}: manifest 记了 gate_up_order", man.get("gate_up_order") == "gate_first")
    check(f"{layout}: manifest 记了种子方案", "100003" in str(man.get("sign_seed_scheme")))
    check(f"{layout}: manifest 记了 top_k", man.get("top_k") == TOPK)
    s = man["layers"]["0"]["w13_snr"]
    check(f"{layout}: 3bit w13 SNR 在合理区间 (p50 {s['p50']:.2f} dB)", 6.0 < s["p50"] < 25.0, str(s))
    nrec = len(man["layers"]["0"]["experts"] or [])
    check(f"{layout}: --full-manifest 记了 {E} 个专家", nrec == E, f"{nrec}")


def section_b_extra(tmp: str) -> None:
    print("\n=== B. 守卫 ===")
    src = os.path.join(tmp, "src_efd")
    acts = os.path.join(tmp, "acts")

    p = run("run", "--in", src, "--out", os.path.join(tmp, "dst_ident"),
            "--bits", "2", "--hessian", "identity", "--layers", "0")
    check("--hessian identity 不给 --acts 也能跑", p.returncode == 0, p.stdout[-800:] + p.stderr[-600:])

    p = run("run", "--in", src, "--out", os.path.join(tmp, "dst_noacts"),
            "--bits", "2", "--layers", "0")
    check("不是 identity 又不给 --acts 时拒绝", p.returncode != 0 and "--acts" in (p.stdout + p.stderr))

    p = run("run", "--in", src, "--out", os.path.join(tmp, "dst_efd"),
            "--bits", "3", "--acts", acts, "--layers", "0")
    check("输出目录非空时拒绝", p.returncode != 0 and "非空" in (p.stdout + p.stderr))

    # 死专家 -> ntok=0 -> DEGRADED。常量激活保证恰好 E-top_k 个专家拿不到 token。
    dsrc = os.path.join(tmp, "src_dead")
    ddst = os.path.join(tmp, "dst_dead")
    dacts = os.path.join(tmp, "acts_const")
    make_ckpt(dsrc, "efd")
    make_const_acts(dacts)
    p = run("run", "--in", dsrc, "--out", ddst, "--bits", "3", "--acts", dacts,
            "--layers", "0", "--full-manifest")
    ok = p.returncode == 0
    check("有死专家时仍能跑完", ok, p.stdout[-900:] + p.stderr[-600:])
    if ok:
        with open(os.path.join(ddst, "ptq_manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        lay = man["layers"]["0"]
        recs = lay["experts"]
        dead = [r for r in recs if r.get("ntok") == 0]
        alive = [r for r in recs if r.get("ntok", 0) > 0]
        check(f"恰好 {E - TOPK} 个专家 ntok=0", len(dead) == E - TOPK, str([r.get("ntok") for r in recs]))
        check(f"死专家被记成 DEGRADED (degraded={lay['degraded']})",
              lay["degraded"] == len(dead) and all(r.get("w13_degraded") and r.get("w2_degraded") for r in dead),
              str(dead))
        check("活专家没被记成 DEGRADED",
              all(not r.get("w13_degraded") and not r.get("w2_degraded") for r in alive), str(alive))
        check(f"日志里报了零命中的个数({E - TOPK})", f"零命中 {E - TOPK} 个" in p.stdout,
              [ln for ln in p.stdout.splitlines() if "零命中" in ln])
        check("死专家仍然被量化了(退化成 H=I,不是跳过)",
              all(r["w13_snr"] > 0 for r in dead), str(dead))

    # 同一条路径的函数级直查:手工给一个带 0 的 ntok
    g = torch.Generator().manual_seed(31)
    w13 = (torch.randn(E, 2 * F, D, generator=g) * 0.05).bfloat16()
    w2 = (torch.randn(E, D, F, generator=g) * 0.05).bfloat16()
    H13 = torch.eye(D).expand(E, D, D).clone()
    H2 = torch.eye(F).expand(E, F, F).clone()
    ntok = torch.tensor([10, 0, 10, 0])
    _, _, recs = PTQ.quantize_layer(w13, w2, H13, H2, ntok, 0, 3, 0.01, True)
    deg_idx = sorted(r["expert"] for r in recs if r.get("w13_degraded"))
    check("quantize_layer: ntok=0 的专家被标 degraded", deg_idx == [1, 3], str(deg_idx))
    check("quantize_layer: ntok>0 的专家没被标",
          all(not r.get("w13_degraded") for r in recs if r["expert"] in (0, 2)))

    # 朝向撞车
    amb = os.path.join(tmp, "src_amb")
    g = torch.Generator().manual_seed(1)
    os.makedirs(amb, exist_ok=True)
    pfx = "model.language_model.layers.0.mlp."
    save_file(
        {pfx + "experts.gate_up_proj": torch.randn(E, 2 * F, 2 * F, generator=g).bfloat16(),
         pfx + "experts.down_proj": torch.randn(E, 2 * F, F, generator=g).bfloat16(),
         pfx + "gate.weight": torch.randn(E, 2 * F, generator=g).bfloat16()},
        os.path.join(amb, "model-00001-of-00001.safetensors"), metadata={"format": "pt"},
    )
    with open(os.path.join(amb, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"num_experts": E, "num_experts_per_tok": TOPK}, fh)
    p = run("run", "--in", amb, "--out", os.path.join(tmp, "dst_amb"),
            "--bits", "3", "--hessian", "identity", "--layers", "0")
    check("朝向撞车时拒绝", p.returncode != 0 and "判不出来" in (p.stdout + p.stderr),
          p.stdout[-500:] + p.stderr[-500:])


# ---------------------------------------------------------------- C
def section_c() -> None:
    print("\n=== C. 2bit 三档对比(会影响排期)===")
    g = torch.Generator().manual_seed(17)
    W = (torch.randn(2 * F, D, generator=g) * 0.05).float()
    # 各向异性的合成 Hessian:真实激活 Hessian 的各向异性远强于此,
    # 所以下面 LDLQ 相对无校准的增益是**下界**。
    A = torch.randn(4 * D, D, generator=g)
    A[:, : D // 4] *= 4.0
    H_real = core.damp_hessian((A.T @ A) / A.shape[0], 0.01)
    H_eye = core.damp_hessian(torch.eye(D), 0.01)
    Hd = core.hadamard(core.HAD_BLOCK)
    si = core.rand_signs(D, 1234)
    so = core.rand_signs(2 * F, 5678)

    rows = []
    for bits in (2, 3):
        cb = core.LinearCodebook(core.D_GROUP, bits=bits)
        rtn = core.rtn_group_qdq(W, bits, 128)
        rows.append((bits, "RTN group128", core.snr_db(W, rtn), core.proxy_loss(W, rtn, H_real)))
        nol = core.unisvq_qdq(W, H_eye, si, so, Hd, cb, use_ldlq=False)
        rows.append((bits, "RHT+码本 (H=I, 无LDLQ)", core.snr_db(W, nol), core.proxy_loss(W, nol, H_real)))
        eye = core.unisvq_qdq(W, H_eye, si, so, Hd, cb, use_ldlq=True)
        rows.append((bits, "RHT+LDLQ (H=I)", core.snr_db(W, eye), core.proxy_loss(W, eye, H_real)))
        real = core.unisvq_qdq(W, H_real, si, so, Hd, cb, use_ldlq=True)
        rows.append((bits, "RHT+LDLQ (真 H)", core.snr_db(W, real), core.proxy_loss(W, real, H_real)))

    print(f"  {'bits':>4}  {'方法':<24} {'SNR (dB)':>9}  {'proxy_loss':>12}")
    for bits, name, snr, pl in rows:
        print(f"  {bits:>4}  {name:<24} {snr:>9.2f}  {pl:>12.4e}")

    by = {(b, n): (s, p) for b, n, s, p in rows}
    for bits in (2, 3):
        rtn_p = by[(bits, "RTN group128")][1]
        eye_p = by[(bits, "RHT+LDLQ (H=I)")][1]
        real_p = by[(bits, "RHT+LDLQ (真 H)")][1]
        nol_p = by[(bits, "RHT+码本 (H=I, 无LDLQ)")][1]
        check(f"{bits}bit: 无校准的 RHT+LDLQ 代理损失低于 RTN", eye_p < rtn_p, f"{eye_p:.3e} vs {rtn_p:.3e}")
        check(f"{bits}bit: 真 H 不差于 H=I", real_p <= eye_p * 1.02, f"{real_p:.3e} vs {eye_p:.3e}")
        check(f"{bits}bit: LDLQ 不差于只有码本", eye_p <= nol_p * 1.02, f"{eye_p:.3e} vs {nol_p:.3e}")
    print("\n  读这张表的前提:权重和 Hessian 都是合成的。所以这是**管路正确性**,")
    print("  不是精度预测;真实激活 Hessian 各向异性更强,LDLQ 与真 H 的增益都会更大。")


def main() -> int:
    print("================ offline_ptq 自测 ================")
    section_a()
    with tempfile.TemporaryDirectory() as tmp:
        make_acts(os.path.join(tmp, "acts"))
        section_b(tmp, "efd")
        section_b(tmp, "edf")
        section_b_extra(tmp)
    section_c()
    print(f"\n================ {_ok} 项通过, {_fail} 项失败 ================")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
