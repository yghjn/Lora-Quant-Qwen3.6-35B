#!/usr/bin/env python3
"""unisvq_core 的 CPU 自测。不需要 NPU、不需要模型权重、不需要 torchtitan。

验的是量化内核本身在 **Qwen3.6-35B-A3B 的真实矩阵形状** 上成立:
  1. Hadamard 正交、RHT 往返是恒等(这是 §风险登记里"静默产生垃圾"的那一类)
  2. 码字级数 == 2^bits,位宽真的接上了(原版 core.py 把 N_BITS 写死成 2)
  3. 阻尼让接近奇异的 H 能过 Cholesky
  4. LDLQ 的代理损失优于同位宽的 RTN
  5. SNR 随位宽单调上升,且 2bit 落在 DSv4.1 实测的量级
  6. w2 的输入维 512 这个"小矩阵"不炸

跑法:  python3 -I test_core_cpu.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import unisvq_core as C  # noqa: E402

torch.manual_seed(0)
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK ' if cond else 'X  '} {name}{(': ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def make_hessian(in_f: int, ntok: int, device="cpu") -> torch.Tensor:
    """H = X^T X,X 是各向异性的激活。ntok < in_f 时 H 接近奇异,正是实战情形
    (冷专家只拿到几十个 token,而输入维是 512/2048)。"""
    scale = torch.logspace(0, -1.5, in_f, device=device)
    X = torch.randn(ntok, in_f, device=device) * scale
    return X.T @ X


# ---------------------------------------------------------------- 1. RHT
print("\n[1] RHT 可逆性")
Hd = C.hadamard(C.HAD_BLOCK)
eye_err = (Hd @ Hd.T - torch.eye(C.HAD_BLOCK)).abs().max().item()
check("Hadamard 正交", eye_err < 1e-5, f"max|H H^T - I| = {eye_err:.2e}")

for out_f, in_f in [(1024, 2048), (2048, 512)]:
    W = torch.randn(out_f, in_f)
    si = C.rand_signs(in_f, seed=1234)
    so = C.rand_signs(out_f, seed=5678)
    Wr = C.block_hadamard(C.block_hadamard(W.T, Hd, so).T, Hd, si)
    Wb = C.block_hadamard_inv(C.block_hadamard_inv(Wr, Hd, si).T, Hd, so).T
    err = (W - Wb).abs().max().item()
    check(f"往返恒等 [{out_f},{in_f}]", err < 1e-4, f"max|W - W''| = {err:.2e}")

# 符号 seed 必须确定性 —— 量化侧和加载侧对不上就静默出垃圾
check(
    "rand_signs 确定性",
    torch.equal(C.rand_signs(512, seed=7), C.rand_signs(512, seed=7)),
    "同 seed 两次调用相同",
)
check(
    "rand_signs 随 seed 变化",
    not torch.equal(C.rand_signs(512, seed=7), C.rand_signs(512, seed=8)),
)

# ---------------------------------------------------------------- 2. 位宽
print("\n[2] 码本位宽接线(原版写死 2bit)")
for bits in (2, 3, 4):
    cb = C.LinearCodebook(d=C.D_GROUP, bits=bits)
    x = torch.randn(4096, 1, C.D_GROUP) * 3.0
    q = cb.quantize(x)
    n_lvl = len(torch.unique(torch.round(q / cb.s + cb.b)))
    check(f"bits={bits} 级数", n_lvl == 2**bits, f"实际 {n_lvl} 档,期望 {2**bits}")

old = os.environ.get("UNISVQ_NBITS")
check("N_BITS 读环境变量", C.N_BITS == int(old) if old else C.N_BITS == 2, f"N_BITS={C.N_BITS}")

# ---------------------------------------------------------------- 3. 阻尼
print("\n[3] Hessian 阻尼(w2 的输入维只有 512,H 常接近奇异)")
H_sing = make_hessian(512, ntok=64)  # 64 个 token / 512 维 -> 必然奇异
try:
    torch.linalg.cholesky(H_sing)
    raw_ok = True
except Exception:
    raw_ok = False
try:
    torch.linalg.cholesky(C.damp_hessian(H_sing))
    damped_ok = True
except Exception:
    damped_ok = False
check("未阻尼的奇异 H 过不了 Cholesky", not raw_ok, "(这正是要阻尼的理由)")
check("阻尼后能过 Cholesky", damped_ok)

# ---------------------------------------------------------------- 4/5/6
print("\n[4,5,6] 真实形状上的量化质量  (Qwen3.6-35B-A3B)")
print(f"      {'矩阵':<18}{'bits':>5}{'SNR(dB)':>10}{'LDLQ 代理':>14}{'RTN 代理':>14}{'增益':>8}{'秒':>7}")
shapes = [("w13 [gate;up]", 1024, 2048), ("w2", 2048, 512)]
snr_table: dict[tuple[str, int], float] = {}

for tag, out_f, in_f in shapes:
    W = torch.randn(out_f, in_f) * 0.02
    H = C.damp_hessian(make_hessian(in_f, ntok=max(4 * in_f, 2048)))
    si = C.rand_signs(in_f, seed=11)
    so = C.rand_signs(out_f, seed=22)
    for bits in (2, 3):
        cb = C.LinearCodebook(d=C.D_GROUP, bits=bits)
        t0 = time.time()
        Wh = C.unisvq_qdq(W, H, si, so, Hd, cb, use_ldlq=True)
        dt = time.time() - t0
        Wr = C.unisvq_qdq(W, H, si, so, Hd, cb, use_ldlq=False)
        pl_l = C.proxy_loss(W, Wh, H)
        pl_r = C.proxy_loss(W, Wr, H)
        snr = C.snr_db(W, Wh)
        snr_table[(tag, bits)] = snr
        print(
            f"      {tag:<18}{bits:>5}{snr:>10.2f}{pl_l:>14.4g}{pl_r:>14.4g}"
            f"{(pl_r / max(pl_l, 1e-30)):>7.2f}x{dt:>7.1f}"
        )
        check(f"{tag} b{bits}: LDLQ 优于 RTN", pl_l < pl_r, f"{pl_l:.4g} < {pl_r:.4g}")
        check(f"{tag} b{bits}: Ŵ 有限", bool(torch.isfinite(Wh).all()))

    check(
        f"{tag}: SNR 随位宽上升",
        snr_table[(tag, 3)] > snr_table[(tag, 2)],
        f"2bit {snr_table[(tag, 2)]:.2f} -> 3bit {snr_table[(tag, 3)]:.2f} dB",
    )

# 和 DSv4.1 实测对账:它的 2bit 重建 SNR 在 7~8 dB 量级
for tag, _, _ in shapes:
    s = snr_table[(tag, 2)]
    check(f"{tag}: 2bit SNR 在合理量级", 2.0 < s < 20.0, f"{s:.2f} dB(DSv4.1 实测 7~8)")

# ---------------------------------------------------------------- 7. RTN 脚手架
print("\n[7] RTN 脚手架(不需要 Hessian)")
W = torch.randn(2048, 512) * 0.02
for bits in (2, 3, 4):
    Wq = C.rtn_group_qdq(W, bits=bits, group=128)
    print(f"      bits={bits}  SNR={C.snr_db(W, Wq):>6.2f} dB")
check(
    "RTN 的 SNR 随位宽上升",
    C.snr_db(W, C.rtn_group_qdq(W, 3)) > C.snr_db(W, C.rtn_group_qdq(W, 2)),
)
try:
    C.rtn_group_qdq(W, 2, group=96)  # 512 不被 96 整除
    divisible_guard = False
except ValueError:
    divisible_guard = True
check("RTN 的整除性守卫会抛", divisible_guard, "group=96 对 in_f=512 被拒绝")

# ---------------------------------------------------------------- 8. int8 容器
print("\n[8] int8 容器(本路线默认不用,保留给 W8A8 部署)")
q, sc = C.requant_int8_perchannel(W)
Wd = q.float() * sc
rel = ((W - Wd).norm() / W.norm()).item()
check("int8 相对误差 < 2%", rel < 0.02, f"{rel * 100:.2f}%(DSv4.1 实测约 0.8%)")
check("int8 码在 [-127,127]", int(q.abs().max()) <= 127)

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 62)
if FAILED:
    print(f"失败 {len(FAILED)} 项:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过。量化内核在 Qwen3.6 的真实矩阵形状上成立。")
sys.exit(0)
