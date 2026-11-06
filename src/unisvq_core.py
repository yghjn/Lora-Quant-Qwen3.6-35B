"""UniSVQ 低位宽 PTQ 内核 —— 纯 torch,无自定义 kernel,无 NPU 依赖。

来源与改动
----------
照搬 `/home/ma-user/work/l84414662/qwen36/quant/core.py`(140 行,UniSVQ 的
独立 CPU 参考实现)。本文件是 **vendoring 副本**,因为目标环境不会有那个工作区。

相对原版只改两处:
  1. `N_BITS` 从模块常量改成读 `UNISVQ_NBITS` 环境变量(原版写死 2,
     会让验证脚本和生产跑的位宽不一致 —— 这是 `l84414662/qwen36/QWEN36_ADAPTATION.md` §1.3 点出的问题)。
  2. `LinearCodebook` 的 `bits` 默认跟随 `N_BITS`,并在 `unisvq_qdq` 里
     不再隐含位宽 —— 位宽只由传进来的 codebook 决定,避免两处不一致。

语义(对齐 lc_qat_torch)
-----------------------
  * RHT:分块 Hadamard(128)+ 随机符号,输入/输出两侧都做
  * codebook:s * (int_lattice - b) @ G^T,G 正交(默认 I),s 按格点方差匹配
  * LDLQ:按 d=4 的列组从后往前量化,用 H 的 LDL 分解回补未量化列

**刻意不包含打包(pack/unpack)。** 假量化路线直接写出 bf16 的 Ŵ,不需要紧凑
产物;而这套软件栈上没有 2bit/3bit 的 matmul kernel,紧凑格式也跑不起来。
要做真正的低位宽部署产物时再加。
"""

from __future__ import annotations

import math
import os

import torch

D_GROUP = 4  # 向量维度 d
N_BITS = int(os.environ.get("UNISVQ_NBITS", "2"))
HAD_BLOCK = 128


# ---------------------------------------------------------------- Hadamard
def _sylvester(n: int, device, dtype=torch.float32) -> torch.Tensor:
    assert n & (n - 1) == 0, f"{n} not a power of 2"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def hadamard(block: int = HAD_BLOCK, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """归一化的 Sylvester-Hadamard 矩阵,正交:H @ H.T == I。"""
    return _sylvester(block, device, dtype)


def block_hadamard(x: torch.Tensor, H: torch.Tensor, sign: torch.Tensor | None = None) -> torch.Tensor:
    """对最后一维分块做正交 Hadamard;sign 为 ±1 向量,长度等于最后一维。"""
    if sign is not None:
        x = x * sign
    blk = H.shape[0]
    *lead, n = x.shape
    assert n % blk == 0, f"最后一维 {n} 不被 HAD_BLOCK={blk} 整除"
    return (x.reshape(*lead, n // blk, blk) @ H).reshape(*lead, n)


def block_hadamard_inv(x: torch.Tensor, H: torch.Tensor, sign: torch.Tensor | None = None) -> torch.Tensor:
    """block_hadamard 的逆。H 正交所以用 H.T;符号在**变换之后**再乘回。"""
    blk = H.shape[0]
    *lead, n = x.shape
    assert n % blk == 0, f"最后一维 {n} 不被 HAD_BLOCK={blk} 整除"
    x = (x.reshape(*lead, n // blk, blk) @ H.T).reshape(*lead, n)
    if sign is not None:
        x = x * sign
    return x


def rand_signs(n: int, seed: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """确定性的 ±1 符号向量。

    **量化侧和加载侧必须用同一个 seed 方案** —— 这是 RHT 可逆性的全部依赖,
    对不上会产出形状正确、内容全错的权重,而那种错只表现为"精度没救回来"。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * 2 - 1
    return s.to(device=device, dtype=dtype)


# ---------------------------------------------------------------- codebook
class LinearCodebook:
    """C_i = s * G @ (w_int_i - b),  w_int in {0..2^bits-1}^d"""

    def __init__(self, d: int = D_GROUP, bits: int | None = None, G=None, device="cpu", dtype=torch.float32):
        bits = N_BITS if bits is None else bits
        if not 1 <= bits <= 8:
            raise ValueError(f"bits 必须在 1..8,得到 {bits}")
        self.d, self.bits = d, bits
        n_lvl = 2**bits
        self.b = (n_lvl - 1) / 2.0
        var = (torch.arange(n_lvl, dtype=torch.float64) ** 2).mean().item() - self.b**2
        self.s = 1.0 / math.sqrt(var)  # 格点方差 -> 1
        self.G = torch.eye(d, device=device, dtype=dtype) if G is None else G.to(device, dtype)
        self.is_identity = G is None
        self.n_lvl = n_lvl
        if self.is_identity:
            # G=I 走逐元素快路径,不需要物化 n_lvl^d 个格点
            # (bits=8, d=4 时那是 42 亿个,必然 OOM)
            self.grid = None
            self.grid_sqnorm = None
        else:
            lattice = torch.cartesian_prod(*([torch.arange(n_lvl, dtype=dtype)] * d)).to(device)
            self.grid = self.s * (lattice - self.b) @ self.G.T  # [n_lvl^d, d]
            self.grid_sqnorm = (self.grid**2).sum(-1)

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        """w: [..., d] -> 最近码字 [..., d]。G=I 时走逐元素舍入的快路径。"""
        if self.is_identity:
            q = torch.round(w / self.s + self.b).clamp_(0, self.n_lvl - 1)
            return self.s * (q - self.b)
        flat = w.reshape(-1, self.d)
        idx = (2.0 * flat @ self.grid.T - self.grid_sqnorm).argmax(-1)
        return self.grid[idx].reshape(w.shape)


# ---------------------------------------------------------------- LDLQ
def block_ldl(H: torch.Tensor, blk: int = D_GROUP):
    """H = L D L^T,L 单位下三角。返回 L(已按 blk 分块置零对角块内上三角)。"""
    n = H.shape[-1]
    Lc = torch.linalg.cholesky(H)
    dblk = Lc.reshape(n // blk, blk, n // blk, blk).diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # [nb,blk,blk]
    DL = torch.linalg.solve_triangular(
        dblk, torch.eye(blk, device=H.device, dtype=H.dtype).expand_as(dblk), upper=False
    )
    L = Lc.reshape(n // blk, blk, n // blk, blk).permute(0, 2, 1, 3)  # [nb, nb, blk, blk]
    L = (L @ DL.unsqueeze(0)).permute(0, 2, 1, 3).reshape(n, n)
    # 对角块置为单位阵、上三角清零
    idx = torch.arange(n // blk, device=H.device)
    Lb = L.reshape(n // blk, blk, n // blk, blk)
    Lb[idx, :, idx, :] = torch.eye(blk, device=H.device, dtype=H.dtype)
    L = Lb.reshape(n, n).tril()
    return L


def ldlq(W: torch.Tensor, H: torch.Tensor, cb: LinearCodebook, blk: int = D_GROUP) -> torch.Tensor:
    """W: [m, n] (n=输入维), H: [n, n]。按列组从后往前量化 + 回补。"""
    m, n = W.shape
    L = block_ldl(H, blk)
    hatW = torch.zeros_like(W)
    err = torch.zeros_like(W)  # W - hatW,已量化部分
    nb = n // blk
    for i in range(nb - 1, -1, -1):
        sl = slice(i * blk, (i + 1) * blk)
        tgt = W[:, sl] + err[:, (i + 1) * blk :] @ L[(i + 1) * blk :, sl]
        q = cb.quantize(tgt.reshape(m, 1, blk)).reshape(m, blk)
        hatW[:, sl] = q
        # QuIP# 传播的是 原始W - 量化值,不是补偿后 target - 量化值
        err[:, sl] = W[:, sl] - q
    return hatW


def damp_hessian(H: torch.Tensor, frac: float = 0.01) -> torch.Tensor:
    """Tikhonov 阻尼 H + frac*mean(diag(H))*I。

    小矩阵上这一项影响更大:Qwen3.6 的 w2 输入维只有 512,H 是 512x512。
    不阻尼时 Cholesky 容易失败(H 由有限样本累出来,常常接近奇异)。
    """
    n = H.shape[-1]
    d = torch.diagonal(H).mean()
    return H + frac * d * torch.eye(n, device=H.device, dtype=H.dtype)


# ---------------------------------------------------------------- 全流程
@torch.no_grad()
def unisvq_qdq(
    W: torch.Tensor,
    H: torch.Tensor,
    sign_in: torch.Tensor | None,
    sign_out: torch.Tensor | None,
    Hd: torch.Tensor,
    cb: LinearCodebook,
    use_ldlq: bool = True,
) -> torch.Tensor:
    """W: [out, in] fp32。返回 Ŵ (fp32, 同形状) —— 已还原回原空间。

    位宽只由 `cb` 决定。H 要先 `damp_hessian` 过。
    """
    out_f, in_f = W.shape
    # 1) RHT:输出侧(行)+ 输入侧(列)
    Wr = block_hadamard(W.T, Hd, sign_out).T  # 行方向
    Wr = block_hadamard(Wr, Hd, sign_in)  # 列方向
    # Hessian 同步右变换
    Hr = block_hadamard(block_hadamard(H, Hd, sign_in).T, Hd, sign_in).T
    # 2) 方差归一到 1(格点是单位方差的)
    sigma = Wr.std().clamp_min(1e-12)
    Wr = Wr / sigma
    # 3) LDLQ / RTN
    Wq = ldlq(Wr, Hr, cb) if use_ldlq else cb.quantize(Wr.reshape(out_f, -1, cb.d)).reshape(out_f, in_f)
    Wq = Wq * sigma
    # 4) 逆 RHT。`.T` 的位置照抄原版,别"顺手"化简 —— 两次转置不对称。
    Wh = block_hadamard_inv(Wq, Hd, sign_in)
    Wh = block_hadamard_inv(Wh.T, Hd, sign_out).T
    return Wh


@torch.no_grad()
def rtn_group_qdq(W: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """逐组对称 int-N 量化再反量化(RTN)。不需要 Hessian,不需要校准。

    只作**管道脚手架**和 LDLQ 的对照下界:2bit 下 RTN 没有实用意义
    (DSv4.1 用了完整 RHT+LDLQ+码本,2bit 裸版 GPQA 仍从 78.28 掉到 48.48)。
    W: [out, in],沿输入维按 group 分组。
    """
    out_f, in_f = W.shape
    if in_f % group != 0:
        raise ValueError(f"输入维 {in_f} 不被 group={group} 整除")
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        raise ValueError(f"bits={bits} 太小")
    w = W.reshape(out_f, in_f // group, group).float()
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.round(w / scale).clamp_(-qmax, qmax)
    return (q * scale).reshape(out_f, in_f).to(W.dtype)


@torch.no_grad()
def requant_int8_perchannel(W: torch.Tensor, eps: float = 1e-12):
    """把 bf16/fp32 权重压回 W8A8_DYNAMIC 容器:int8 + per-output-channel scale。

    本路线默认**不用**(基座保持 bf16)。保留它是为了:最终部署形态若确定走
    W8A8 容器,在合并之后加一次这个即可,不影响前面任何步骤。
    """
    amax = W.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = amax / 127.0
    q = torch.round(W / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


def proxy_loss(W: torch.Tensor, Wh: torch.Tensor, H: torch.Tensor) -> float:
    """QuIP 代理损失 tr((W-Ŵ) H (W-Ŵ)^T)。量化质量的主判据。"""
    E = (W - Wh).float()
    return torch.einsum("mi,ij,mj->", E, H.float(), E).item()


def snr_db(W: torch.Tensor, Wh: torch.Tensor) -> float:
    """重建信噪比 (dB)。DSv4.1 的 2bit 实测在 7~8 dB 量级。"""
    num = (W.float() ** 2).sum()
    den = ((W - Wh).float() ** 2).sum().clamp_min(1e-30)
    return float(10.0 * torch.log10(num / den))
