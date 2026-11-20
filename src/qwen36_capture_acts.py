#!/usr/bin/env python
"""抓各层 MoE 输入的 hidden states,给 `offline_ptq.py --acts` 用(工单 §2.2 一直缺的那一段)。

为什么要它
----------
`offline_ptq.py` 带 Hessian 的量化(RHT + LDLQ + 码本,DSv4.1 用的那一类)要每层 MoE
的输入激活。2026-11-20 之前本仓库没有抓激活的代码,所以 3bit 一直是直接取整(RTN)。

怎么抓:不另起一套前向,**挂在已经跑通的训练进程上**
------------------------------------------------------
`CONFIG=qwen36_35b_a3b_capture_acts`(`qwen36_lora_recipe.py`)和正式训练走完全同一条
路:同样的权重加载、同样的 8 卡并行、同样的语料。区别只有两处:

* 学习率是 0。LoRA 的 B 矩阵初值是全零,学习率 0 它就一直是零,adapter 的贡献恒为 0,
  模型**严格等于**加载进来的那份权重。本模块每一步前向之前都会核对 B 仍是全零,
  不是就立刻抛异常。
* 不保存 adapter,改成在每层 MoE 模块上挂钩子,把它的输入存下来。

所以 `HF_ASSETS_PATH` 必须指向**原始权重**(不是量化后的目录;recipe 会拦)。

存成什么
--------
`$CAPTURE_ACTS_DIR/layer<NN>/r<卡号>_s<步>_n<token 数>.pt`,每个文件是一张卡一步的
`[n, dim]` bf16 张量。`offline_ptq.py` 和 `mp_select.py` 的 `--acts` 直接认这个目录。

* 行尾的补齐位置不存。上游把样本拼满一行,放不下一条时用结束符把行尾补齐,
  那些位置不是真数据。判法见 `keep_mask`。
* activation checkpoint 会在反向时把每层再算一遍,钩子会被再叫一次 —— 按"这一次前向
  这一层存过没有"去重。
* 另存一小份 `probe/layer<NN>.pt`(0 号卡、第一步、每层前 64 个 token):专家模块的
  输入、路由打分、路由到的专家号、输出。`check` 子命令拿它和 checkpoint 里的权重在
  CPU 上重算一遍对拍,用来确认 `--gate-up-order` —— 那个开关弄反会静默出错。

环境变量
--------
    CAPTURE_ACTS_DIR   输出目录,绝对路径,必填
    CAPTURE_TOKENS     每层大约抓多少个 token 位置,默认 100 万。步数 = 它 ÷ (卡数 × seq_len),
                       向上取整。磁盘占用 = 位置数 × dim × 2 字节 × 层数:
                       100 万位置、dim 2048、40 层约 164 GB(去掉补齐后略少)
    CAPTURE_LAYERS     只抓哪些层,如 20 / 0-3 / 0,5,9;默认全部。单层试跑用

抓多少够:DSv4.1 的经验是看**每个专家**分到的 token 数相对输入维度的倍数,不是总量
(它 157 万 token 时每专家约 6 倍输入维,加到 1004 万后约 38 倍,2bit 下有一项评测涨了
8 分;再往上加没有变化)。Qwen3.6 每个 token 进 8/256 个专家,100 万 token 是每专家平均
约 3 万,输入维 2048 的 15 倍左右。

核对产物(纯 CPU,不占卡)
--------------------------
    python -I src/qwen36_capture_acts.py check --acts <目录> --in <原始权重目录>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from functools import partial

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen36_adapter_save import canonical_name  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "install", "find_modules", "planned_steps", "keep_mask", "scan", "probe_errors",
    "refuse_quantized_source", "require_out_dir",
]

ENV_DIR = "CAPTURE_ACTS_DIR"
ENV_TOKENS = "CAPTURE_TOKENS"
ENV_LAYERS = "CAPTURE_LAYERS"
DEFAULT_TOKENS = 1_000_000
PROBE_TOKENS = 64
META_NAME = "capture_meta.json"
PROBE_DIR = "probe"

# 量化 / 合并工具在产物目录里留下的记录文件。有它们就说明这不是原始权重。
QUANTIZED_MARKERS = ("fakequant_manifest.json", "ptq_manifest.json", "merge_info.json")

_MOE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.moe$")
_ROUTED_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.moe\.routed_experts$")
_LAYER_DIR_RE = re.compile(r"^layer0*(\d+)$")
_PART_RE = re.compile(r"^r(\d+)_s(\d+)_n(\d+)\.pt$")


# ---------------------------------------------------------------- 小工具
def layer_dir(layer: int) -> str:
    return f"layer{layer:02d}"


def part_name(rank: int, step: int, n: int) -> str:
    return f"r{rank:02d}_s{step:05d}_n{n}.pt"


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _local(t):
    """DTensor 取本卡那一片;普通张量原样返回。"""
    return t.to_local() if hasattr(t, "to_local") else t


def env_tokens() -> int:
    raw = (os.environ.get(ENV_TOKENS) or "").strip()
    if not raw:
        return DEFAULT_TOKENS
    try:
        value = int(float(raw))
    except ValueError:
        raise RuntimeError(f"{ENV_TOKENS} 必须是一个正数(现在是 {raw!r})") from None
    if value <= 0:
        raise RuntimeError(f"{ENV_TOKENS} 必须是一个正数(现在是 {raw!r})")
    return value


def planned_steps(seq_len: int, world: int, tokens: int | None = None) -> int:
    """抓够 `tokens` 个位置要跑多少步。一步是每张卡各一行 seq_len。"""
    tokens = env_tokens() if tokens is None else tokens
    return max(1, math.ceil(tokens / (seq_len * world)))


def bytes_needed(steps: int, world: int, seq_len: int, dim: int, n_layers: int) -> int:
    """上限:按一个位置都不去掉算。bf16 每个数 2 字节。"""
    return steps * world * seq_len * dim * 2 * n_layers


def parse_layers(spec: str | None, available: list[int]) -> list[int]:
    if not spec or spec.strip() in ("", "all"):
        return sorted(available)
    out: list[int] = []
    try:
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
    except ValueError:
        raise RuntimeError(f"{ENV_LAYERS} 看不懂:{spec!r}(写法如 20 / 0-3 / 0,5,9)") from None
    missing = sorted(set(out) - set(available))
    if missing or not out:
        raise RuntimeError(f"{ENV_LAYERS}={spec!r}:模型里没有这些层 {missing};可用 {min(available)}..{max(available)}")
    return sorted(set(out))


def require_out_dir(out_dir: str | None = None) -> str:
    out_dir = (out_dir or os.environ.get(ENV_DIR) or "").strip()
    if not out_dir:
        raise RuntimeError(f"{ENV_DIR} 未设置 —— 它是激活的输出目录。不设的话这一趟什么都不会留下。")
    if not os.path.isabs(out_dir):
        raise RuntimeError(
            f"{ENV_DIR} 必须是绝对路径(现在是 {out_dir!r})—— "
            "训练是 cd 到 torchtitan-npu 目录里起的,相对路径会指到别处。"
        )
    return out_dir


def refuse_quantized_source(assets: str) -> None:
    """校准激活必须来自原始权重。量化后的目录里有工具留下的记录文件,据此拦住。"""
    for name in QUANTIZED_MARKERS:
        if assets and os.path.isfile(os.path.join(assets, name)):
            raise RuntimeError(
                f"HF_ASSETS_PATH={assets} 里有 {name} —— 这是量化或合并之后的目录。"
                "抓激活要用**原始权重**:量化器要的是没量化的模型里各层实际看到的输入。"
            )


def keep_mask(tokens: torch.Tensor) -> torch.Tensor:
    """`[B, L]` 的 bool,True = 真数据,False = 行尾补齐。

    上游 `ChatDataset` 把样本一条接一条拼满一行;下一条放不下时,用结束符把这一行
    剩下的位置补齐。每条样本自己的最后一个结束符不在输入里(输入是 `tokens[:-1]`),
    所以**行尾连续相同的那一串就是补齐**。只有一个时分不出来,留着 —— 4096 里多一个
    位置没有影响。
    """
    t = _local(tokens).detach().to("cpu")
    if t.dim() == 1:
        t = t.unsqueeze(0)
    n_rows, length = t.shape
    keep = torch.ones(n_rows, length, dtype=torch.bool)
    for b in range(n_rows):
        row = t[b]
        differs = (row != row[-1]).nonzero()
        run = length - (int(differs[-1]) + 1) if differs.numel() else length
        if run >= 2:
            keep[b, length - run:] = False
    return keep


def _atomic_save(obj, path: str) -> None:
    """先写 .tmp 再改名:进程中途被杀时不会留下一个读得出来但不完整的文件。"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 训练进程里的钩子
class _Capture:
    def __init__(self, out_dir: str, layers: list[int], limit: int, rank: int, lora_b: list):
        self.out_dir = out_dir
        self.layers = set(layers)
        self.limit = limit
        self.rank = rank
        self.lora_b = lora_b
        self.fwd = 0
        self.keep: torch.Tensor | None = None
        self.seen: set[int] = set()
        self.probe_left = set(layers) if rank == 0 else set()
        self.kept_total = 0
        self.t0 = time.time()

    # -- 每次前向开头:整个模型的输入(token id)
    def on_model_input(self, _module, args, kwargs):
        self.fwd += 1
        self.seen = set()
        self.keep = None
        if self.fwd > self.limit:
            if self.fwd == self.limit + 1 and self.rank == 0:
                logger.info("抓激活: 计划的 %d 步已抓完,之后的前向不再存", self.limit)
            return None
        self._assert_base_model()
        tokens = args[0] if args else kwargs.get("tokens")
        tokens = _local(tokens) if tokens is not None else None
        if tokens is None or tokens.dim() != 2 or tokens.is_floating_point():
            raise RuntimeError(
                "抓激活: 模型的第一个输入不是 [B, L] 的 token id —— 行尾补齐没法判。"
                f"拿到的是 {None if tokens is None else (tuple(tokens.shape), tokens.dtype)}"
            )
        self.keep = keep_mask(tokens)
        return None

    def _assert_base_model(self) -> None:
        """LoRA 的 B 必须还是全零,否则抓到的不是原始模型的激活。"""
        if not self.lora_b:
            return
        with torch.no_grad():
            worst = torch.stack([_local(p).detach().abs().max().float() for p in self.lora_b]).max().item()
        if worst != 0.0:
            raise RuntimeError(
                f"抓激活: LoRA 的 B 矩阵不是全零(最大绝对值 {worst:.3e})—— 模型已经不等于"
                "加载进来的权重。这个配置的学习率应当是 0;检查是不是用错了 CONFIG。"
            )

    # -- 每层 MoE 的输入
    def on_moe_input(self, layer: int, _module, args, kwargs):
        if self.keep is None or layer in self.seen:
            return None
        self.seen.add(layer)
        x = _local(args[0] if args else kwargs.get("x_BLD"))
        if x.dim() != 3 or tuple(x.shape[:2]) != tuple(self.keep.shape):
            raise RuntimeError(
                f"抓激活: 第 {layer} 层 MoE 输入形状 {tuple(x.shape)},"
                f"和这一步的 token 形状 {tuple(self.keep.shape)} 对不上"
            )
        flat = x.detach().to(torch.bfloat16).cpu().reshape(-1, x.shape[-1])
        flat = flat[self.keep.reshape(-1)].contiguous()
        n = int(flat.shape[0])
        if n:
            if not bool(torch.isfinite(flat.float()).all()):
                raise RuntimeError(f"抓激活: 第 {layer} 层第 {self.fwd} 步的 MoE 输入里有 NaN/Inf")
            d = os.path.join(self.out_dir, layer_dir(layer))
            os.makedirs(d, exist_ok=True)
            _atomic_save(flat, os.path.join(d, part_name(self.rank, self.fwd, n)))
        if len(self.seen) == len(self.layers):
            self.kept_total += n
            if self.rank == 0:
                total = self.keep.numel()
                logger.info(
                    "抓激活: 第 %d/%d 步,本卡这一步 %d 个位置留 %d 个(行尾补齐 %.1f%%),%d 层,累计 %.0f 秒",
                    self.fwd, self.limit, total, n, 100.0 * (total - n) / max(total, 1),
                    len(self.layers), time.time() - self.t0,
                )
        return None

    # -- 专家模块的输入和输出,只存一小份,给 check 对拍
    def on_routed(self, layer: int, _module, args, kwargs, output):
        if layer not in self.probe_left or self.keep is None:
            return None
        self.probe_left.discard(layer)
        try:
            names = ("x_BLD", "topk_scores_BLK", "topk_expert_ids_BLK")
            x, scores, ids = (_local(args[i] if i < len(args) else kwargs[n]) for i, n in enumerate(names))
            out = _local(output)
            n = min(PROBE_TOKENS, int(x.shape[1]))
            rec = {
                "x": x[0, :n].detach().float().cpu(),
                "scores": scores[0, :n].detach().float().cpu(),
                "ids": ids[0, :n].detach().long().cpu(),
                "out": out[0, :n].detach().float().cpu(),
            }
            d = os.path.join(self.out_dir, PROBE_DIR)
            os.makedirs(d, exist_ok=True)
            _atomic_save(rec, os.path.join(d, layer_dir(layer) + ".pt"))
        except Exception as exc:  # noqa: BLE001 - 对拍样本是附带的,抓不到不该让整趟白跑
            logger.warning("抓激活: 第 %d 层的对拍样本没存下来(%s: %s)—— 激活本身不受影响", layer, type(exc).__name__, exc)
        return None


def _existing_parts(out_dir: str) -> list[str]:
    found: list[str] = []
    if not os.path.isdir(out_dir):
        return found
    for fn in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, fn)
        if os.path.isdir(full) and (_LAYER_DIR_RE.match(fn) or fn == PROBE_DIR):
            if os.listdir(full):
                found.append(fn)
        elif os.path.isfile(full) and re.match(r"^layer\d+\.pt$", fn):
            found.append(fn)
    return found


def _free_bytes(path: str) -> int:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe or "/").free


def find_modules(root) -> tuple[dict, dict, list, int | None]:
    """在模型里找要挂钩子的地方。返回 ({层: MoE 模块}, {层: 专家模块}, [LoRA B 参数], dim)。

    模块名里由包装模块引入的段(`_checkpoint_wrapped_module` 等)先去掉再认。
    `verify/probe_parallelize_cpu.py` 拿并行化之后的真模型调它,不用上卡就能知道找不找得到。
    """
    moes: dict[int, torch.nn.Module] = {}
    routed: dict[int, torch.nn.Module] = {}
    for name, mod in root.named_modules():
        cname = canonical_name(name)
        m = _MOE_RE.search(cname)
        if m:
            moes[int(m.group(1))] = mod
        m = _ROUTED_RE.search(cname)
        if m:
            routed[int(m.group(1))] = mod
    lora_b = [p for n, p in root.named_parameters() if "lora_b" in n]
    dim = None
    try:
        dim = int(moes[min(moes)].router.gate.weight.shape[-1])
    except Exception:  # noqa: BLE001 - 拿不到维度就不做磁盘预检
        pass
    return moes, routed, lora_b, dim


def install(model_parts, *, steps: int, seq_len: int, world: int,
            out_dir: str | None = None, layers_spec: str | None = None,
            source: str | None = None, corpus: str | None = None) -> int:
    """在模型上挂抓激活的钩子。返回抓多少层。只抓前 `steps` 次前向。"""
    out_dir = require_out_dir(out_dir)
    rank = _rank()
    if len(model_parts) != 1:
        raise RuntimeError(f"抓激活: 只支持模型不分段的情况,现在有 {len(model_parts)} 段")
    root = model_parts[0]

    moes, routed, lora_b, dim = find_modules(root)
    if not moes:
        raise RuntimeError("抓激活: 模型里没找到 layers.<N>.moe —— 模块名变了?")
    layers = parse_layers(layers_spec if layers_spec is not None else os.environ.get(ENV_LAYERS), sorted(moes))

    old = _existing_parts(out_dir)
    if old:
        raise RuntimeError(
            f"抓激活: {out_dir} 里已经有激活({', '.join(old[:4])}{' ...' if len(old) > 4 else ''})。"
            "两趟的结果混在一起分不出来 —— 换一个目录,或者先把它删掉。"
        )

    need = bytes_needed(steps, world, seq_len, dim, len(layers)) if dim else 0
    if need:
        free = _free_bytes(out_dir)
        if need > free * 0.95:
            fit = int(free * 0.9 / (dim * 2 * len(layers)))
            raise RuntimeError(
                f"抓激活: {steps} 步 × {world} 卡 × {seq_len} × {len(layers)} 层要 {need / 2**30:.0f} GiB,"
                f"{out_dir} 所在磁盘只剩 {free / 2**30:.0f} GiB。"
                f"把 {ENV_TOKENS} 调到 {fit} 以下,或者用 {ENV_LAYERS} 分批抓,或者换一块盘。"
            )

    cap = _Capture(out_dir, layers, steps, rank, lora_b)
    root.register_forward_pre_hook(cap.on_model_input, with_kwargs=True)
    for ln in layers:
        moes[ln].register_forward_pre_hook(partial(cap.on_moe_input, ln), with_kwargs=True)
        if ln in routed:
            routed[ln].register_forward_hook(partial(cap.on_routed, ln), with_kwargs=True)

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "source": source,
            "corpus": corpus,
            "steps": steps,
            "world": world,
            "seq_len": seq_len,
            "dim": dim,
            "layers": layers,
            "positions_per_layer_max": steps * world * seq_len,
            "lora_b_tensors_checked": len(lora_b),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(out_dir, META_NAME), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        logger.info(
            "抓激活已注册: %d 层,前 %d 步,每层最多 %d 个位置,磁盘上限约 %.0f GiB -> %s"
            "(每步核对 %d 个 LoRA B 张量仍是全零)",
            len(layers), steps, steps * world * seq_len, need / 2**30, out_dir, len(lora_b),
        )
    return len(layers)


# ---------------------------------------------------------------- 核对产物(CPU)
def scan(acts_dir: str) -> tuple[dict[int, list[tuple[int, int, int, str]]], list[str]]:
    """返回 ({层: [(卡号, 步, token 数, 路径), ...]}, [没写完的 .tmp 文件])。只看文件名,不读数据。"""
    layers: dict[int, list[tuple[int, int, int, str]]] = {}
    leftovers: list[str] = []
    for fn in sorted(os.listdir(acts_dir)):
        m = _LAYER_DIR_RE.match(fn)
        full = os.path.join(acts_dir, fn)
        if not m or not os.path.isdir(full):
            continue
        parts = layers.setdefault(int(m.group(1)), [])
        for pf in sorted(os.listdir(full)):
            pm = _PART_RE.match(pf)
            if pm:
                parts.append((int(pm.group(1)), int(pm.group(2)), int(pm.group(3)), os.path.join(full, pf)))
            elif pf.endswith(".tmp"):
                leftovers.append(os.path.join(full, pf))
    return layers, leftovers


@torch.no_grad()
def probe_errors(x: torch.Tensor, scores: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                 w13: torch.Tensor, w2: torch.Tensor) -> dict[str, float]:
    """用 checkpoint 的权重把专家模块的输出重算一遍,两种 gate/up 顺序各算一次。

    x [n, D]  scores/ids [n, K]  out [n, D]  w13 [E, 2F, D]  w2 [E, D, F](都是 [out, in])。
    模型里是 `out = Σ_k score_k · W2[e_k] · (silu(gate) * up)`,打分乘在专家输出上。
    返回两种顺序各自的相对误差 `||重算 - 实际|| / ||实际||`;对的那个应当接近 0。
    """
    hidden = w13.shape[1] // 2
    est = {"gate_first": torch.zeros_like(out), "up_first": torch.zeros_like(out)}
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            g_u = w13[e].float() @ x[t].float()
            first, second = g_u[:hidden], g_u[hidden:]
            w2e = w2[e].float()
            est["gate_first"][t] += scores[t, k] * (w2e @ (torch.nn.functional.silu(first) * second))
            est["up_first"][t] += scores[t, k] * (w2e @ (torch.nn.functional.silu(second) * first))
    denom = float(out.norm()) + 1e-20
    return {name: float((y - out).norm()) / denom for name, y in est.items()}


def _fmt_gb(nbytes: float) -> str:
    return f"{nbytes / 2**30:.1f} GiB"


def cmd_check(args) -> int:
    acts = args.acts
    if not os.path.isdir(acts):
        print(f"!! 目录不存在: {acts}", file=sys.stderr)
        return 2
    problems: list[str] = []
    notes: list[str] = []
    meta = {}
    meta_path = os.path.join(acts, META_NAME)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    layers, leftovers = scan(acts)
    if not layers or not any(layers.values()):
        print(f"!! {acts} 下没有 layer<NN>/r*_s*_n*.pt —— 抓激活那一步没产出东西", file=sys.stderr)
        return 2

    print("================ 激活目录 ================")
    print(f"目录            {os.path.abspath(acts)}")
    if meta:
        print(f"权重来源        {meta.get('source')}")
        print(f"语料            {meta.get('corpus')}")
        print(f"计划            {meta.get('steps')} 步 × {meta.get('world')} 卡 × seq_len {meta.get('seq_len')}")
    else:
        notes.append(f"没有 {META_NAME},没法核对分片数够不够")
    lns = sorted(layers)
    print(f"层              {len(lns)} 层: {lns[0]}..{lns[-1]}" if len(lns) > 4 else f"层              {lns}")
    if meta.get("layers") and sorted(meta["layers"]) != lns:
        problems.append(f"记录里要抓 {len(meta['layers'])} 层,目录里有 {len(lns)} 层")

    # 各层的 (卡, 步, token 数) 必须一模一样:同一步同一张卡,每层留下的位置是同一批
    ref = [(r, s, n) for r, s, n, _ in layers[lns[0]]]
    for ln in lns[1:]:
        if [(r, s, n) for r, s, n, _ in layers[ln]] != ref:
            problems.append(f"第 {ln} 层的分片和第 {lns[0]} 层不一致(有的层少了分片)")
    n_tok = sum(n for _, _, n in ref)
    ranks = sorted({r for r, _, _ in ref})
    steps = sorted({s for _, s, _ in ref})
    print(f"每层 token      {n_tok:,}   分片 {len(ref)} 个 = {len(steps)} 步 × {len(ranks)} 卡")
    if meta.get("steps") and meta.get("world"):
        want = int(meta["steps"]) * int(meta["world"])
        if len(ref) != want:
            problems.append(f"每层应有 {want} 个分片({meta['steps']} 步 × {meta['world']} 卡),实际 {len(ref)} 个 —— 没抓完")
        if meta.get("seq_len"):
            full = want * int(meta["seq_len"])
            print(f"                去掉行尾补齐 {100.0 * (full - n_tok) / max(full, 1):.1f}%(共 {full:,} 个位置)")
    if leftovers:
        problems.append(f"有 {len(leftovers)} 个没写完的分片(.tmp),例如 {leftovers[0]} —— 抓激活那一步中途断过")

    # 每层抽一个分片真读出来看
    dim = None
    for ln in lns:
        _, _, n, path = layers[ln][0]
        t = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(t, torch.Tensor) or t.dim() != 2 or t.shape[0] != n:
            problems.append(f"第 {ln} 层 {os.path.basename(path)}: 不是 [{n}, dim] 的张量")
            continue
        dim = dim or int(t.shape[1])
        if int(t.shape[1]) != dim:
            problems.append(f"第 {ln} 层的 dim 是 {t.shape[1]},别的层是 {dim}")
        tf = t.float()
        if not bool(torch.isfinite(tf).all()):
            problems.append(f"第 {ln} 层 {os.path.basename(path)}: 有 NaN/Inf")
        elif float(tf.std()) == 0.0:
            problems.append(f"第 {ln} 层 {os.path.basename(path)}: 全是同一个数")
    if dim:
        per_layer = n_tok * dim * 2
        print(f"磁盘            每层 {_fmt_gb(per_layer)},合计 {_fmt_gb(per_layer * len(lns))}")
        print(f"抽查            {len(lns)} 层各读 1 个分片:形状、dim={dim}、没有 NaN/Inf")

    verdict = None
    if args.src:
        import offline_ptq as ptq

        idx = ptq.index_checkpoint(args.src)
        geo = ptq.read_geometry(args.src)
        some = idx[sorted(idx)[0]]
        e_n = some["gate_up"][2][0]
        top_k = args.top_k or geo["top_k"]
        if top_k:
            avg = n_tok * top_k / e_n
            print(f"每专家平均      {avg:,.0f} token(= 每层 token × {top_k} ÷ {e_n}"
                  + (f";输入维 {dim} 的 {avg / dim:.1f} 倍)" if dim else ")"))
            if dim and avg < 4 * dim:
                notes.append(f"每专家平均只有输入维的 {avg / dim:.1f} 倍,偏少(DSv4.1 嫌 6 倍不够);可以加大 {ENV_TOKENS} 再抓")
        if meta.get("source") and os.path.abspath(str(meta["source"])) != os.path.abspath(args.src):
            notes.append(f"激活来自 {meta['source']},而 --in 给的是 {args.src} —— 确认是同一份权重")

        probe_dir = os.path.join(acts, PROBE_DIR)
        have = [ln for ln in lns if os.path.isfile(os.path.join(probe_dir, layer_dir(ln) + ".pt")) and ln in idx]
        if not have:
            notes.append("没有对拍样本(probe/),gate/up 顺序没法在这里确认")
        else:
            want_n = max(1, args.probe_layers)
            pick = sorted({have[round(i * (len(have) - 1) / max(want_n - 1, 1))] for i in range(min(want_n, len(have)))})
            print(f"\n================ 对拍:用 checkpoint 的权重重算专家模块的输出 ================")
            votes: list[str] = []
            for ln in pick:
                rec = torch.load(os.path.join(probe_dir, layer_dir(ln) + ".pt"), map_location="cpu", weights_only=True)
                d = idx[ln]
                gu_shape, dn_shape = d["gate_up"][2], d["down"][2]
                a, b = gu_shape[1], gu_shape[2]
                w_dim = hidden = None
                for d_c, h_c in [(b, a // 2), (a, b // 2)]:
                    if sorted(dn_shape[1:]) == sorted([d_c, h_c]):
                        w_dim, hidden = d_c, h_c
                        break
                if w_dim is None or w_dim != rec["x"].shape[-1]:
                    problems.append(f"第 {ln} 层:权重的 dim 和激活的 dim({rec['x'].shape[-1]})对不上 —— 不是同一个模型")
                    continue
                layout = ptq.decide_layout(gu_shape, dn_shape, w_dim, hidden, args.layout)
                w13 = ptq.as_out_in(ptq.get_tensor(args.src, d["gate_up"][0], d["gate_up"][1]), layout)
                w2 = ptq.as_out_in(ptq.get_tensor(args.src, d["down"][0], d["down"][1]), layout)
                errs = probe_errors(rec["x"], rec["scores"], rec["ids"], rec["out"], w13, w2)
                line = f"  层 {ln:>2}   gate 在前 误差 {100 * errs['gate_first']:.1f}%   up 在前 误差 {100 * errs['up_first']:.1f}%"
                if "router" in d and top_k:
                    mine = ptq.route(rec["x"], ptq.get_tensor(args.src, d["router"][0], d["router"][1]), top_k)
                    same = sum(set(mine[t].tolist()) == set(rec["ids"][t].tolist()) for t in range(mine.shape[0]))
                    line += f"   路由一致 {same}/{mine.shape[0]}"
                    if same < 0.9 * mine.shape[0]:
                        problems.append(f"第 {ln} 层:离线重算的路由和模型里的只有 {same}/{mine.shape[0]} 一致 —— "
                                        "量化器分 token 靠的就是这个重算")
                best = min(errs, key=errs.get)
                other = "up_first" if best == "gate_first" else "gate_first"
                if errs[best] < 0.2 and errs[other] > 2 * errs[best]:
                    votes.append(best)
                    line += f"   -> {best}"
                else:
                    line += "   -> 判不出来"
                print(line)
            if votes and len(set(votes)) == 1 and len(votes) == len(pick):
                verdict = votes[0]
            else:
                problems.append("gate/up 顺序判不出来(两种顺序重算出来都对不上模型的实际输出)—— "
                                "先别量化;确认 --in 是抓激活时用的那份权重")

    print("\n================ 结论 ================")
    for n in notes:
        print(f"注意: {n}")
    for p in problems:
        print(f"!! {p}")
    if problems:
        print("没通过。上面带 !! 的要先解决。")
        return 1
    if verdict:
        extra = "(就是默认值,不用另外传)" if verdict == "gate_first" else "(**不是默认值,量化时必须显式传**)"
        print(f"通过。量化时用 --gate-up-order {verdict}{extra}")
    else:
        print("通过(只核对了文件;加 --in <原始权重目录> 可以连 gate/up 顺序一起对拍)。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="抓激活的产物核对(抓本身在训练进程里,见文件头)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("check", help="核对激活目录:齐不齐、有没有坏数据、gate/up 顺序")
    pc.add_argument("--acts", required=True, help="CAPTURE_ACTS_DIR 那个目录")
    pc.add_argument("--in", dest="src", help="原始权重目录;给了就连 gate/up 顺序和路由一起对拍")
    pc.add_argument("--layout", choices=("auto", "efd", "edf"), default="auto")
    pc.add_argument("--top-k", type=int, help="router top-k;默认从 config.json 读")
    pc.add_argument("--probe-layers", type=int, default=3, help="对拍几层(均匀取,默认 3;每层要读约 1.5 GB 权重)")
    pc.set_defaults(func=cmd_check)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
