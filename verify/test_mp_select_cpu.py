#!/usr/bin/env python
"""mp_select.py 和 offline_ptq.py --keep-hot 的自测(工单 §4.1 / §4.2)。

    python verify/test_mp_select_cpu.py

纯 CPU、不占卡、不需要真权重和真激活。分四段:

**A. 位宽账的数学**(纯函数,不碰文件)
  锚点是 DSv4.1 已发表的那个数:`2bit + hot39/384 保 int8 = 2.609 bit`。
  公式对不上这个数,说明口径错了,后面所有跨路线比较都没意义。

**B. 选择逻辑**(纯函数,手造计数表)
  top-K、并列打破、cold 变体、hot/cold 不相交。

**C. mp_select 端到端**(走子进程)
  最强的那条判据用**常量激活**:每一行激活都相同 -> 路由对所有 token 一致 ->
  恰好 `top_k` 个专家吃到全部 token、`E - top_k` 个拿到 0 个。
  于是 `hot` 列表必须**精确等于**被路由命中的那几个专家号,不是"差不多"。

**D. offline_ptq --keep-hot 端到端**(走子进程)
  保留的专家必须**逐位等于原权重**(没量化),没保留的必须变;
  平均位宽进 manifest 和分片 metadata;三类坏 keep json 必须被拒。

几何沿用 `test_offline_ptq_cpu.py`:`E=4 / dim=384 / hidden=128 / top_k=2`。
`dim` 和 `hidden` 都被 `HAD_BLOCK=128` 整除,且 `dim != 2*hidden`
(否则两种朝向形状一样、判不出来)。

E=4 太小,所以 `-k` 的默认值(DSv4.1 hot39/384 的等比例换算)会算成 0 ——
这不是 bug,守卫会拒。检查 11 就是测它。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import mp_select as MPS  # noqa: E402

SEL = os.path.join(HERE, "..", "src", "mp_select.py")
PTQ_SCRIPT = os.path.join(HERE, "..", "src", "offline_ptq.py")
E, D, F, TOPK = 4, 384, 128, 2
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


def run(script: str, *extra: str) -> subprocess.CompletedProcess:
    argv = [sys.executable] + (["-I"] if sys.flags.isolated else []) + [script, *extra]
    return subprocess.run(argv, capture_output=True, text=True)


# ---------------------------------------------------------------- 夹具
def make_ckpt(dirpath: str) -> dict:
    """efd 朝向的合成 checkpoint。专家 e 永远是第 0 维的下标,两种朝向都一样。"""
    g = torch.Generator().manual_seed(5)
    t: dict[str, torch.Tensor] = {}
    for layer in LAYERS:
        p = f"model.language_model.layers.{layer}.mlp."
        t[p + "experts.gate_up_proj"] = (torch.randn(E, 2 * F, D, generator=g) * 0.05).bfloat16()
        t[p + "experts.down_proj"] = (torch.randn(E, D, F, generator=g) * 0.05).bfloat16()
        t[p + "gate.weight"] = torch.randn(E, D, generator=g).bfloat16()
        t[p + "shared_expert.gate_proj.weight"] = torch.randn(F, D, generator=g).bfloat16()
    os.makedirs(dirpath, exist_ok=True)
    save_file(t, os.path.join(dirpath, "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(dirpath, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "qwen3_5_moe", "num_experts": E,
                   "num_experts_per_tok": TOPK}, fh)
    return t


def make_const_acts(dirpath: str, n: int = 256) -> None:
    os.makedirs(dirpath, exist_ok=True)
    g = torch.Generator().manual_seed(21)
    row = torch.randn(1, D, generator=g)
    for layer in LAYERS:
        torch.save(row.expand(n, D).contiguous().bfloat16(),
                   os.path.join(dirpath, f"layer{layer}.pt"))


def routed_experts(ckpt: dict, layer: int, acts_dir: str) -> list[int]:
    """独立算一遍:这一层哪几个专家被常量激活路由到。不调被测代码的 select。"""
    w = ckpt[f"model.language_model.layers.{layer}.mlp.gate.weight"].float()
    x = torch.load(os.path.join(acts_dir, f"layer{layer}.pt"), weights_only=True).float()
    logits = x[:1] @ w.T
    return sorted(torch.topk(logits[0], k=TOPK).indices.tolist())


def read_dir(dirpath: str) -> dict[str, torch.Tensor]:
    out = {}
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(dirpath, fn), framework="pt") as f:
            for k in f.keys():  # noqa: SIM118
                out[k] = f.get_tensor(k)
    return out


def shard_meta(dirpath: str) -> dict:
    for fn in sorted(os.listdir(dirpath)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(dirpath, fn), framework="pt") as f:
                return f.metadata() or {}
    return {}


# ---------------------------------------------------------------- A
def section_a() -> None:
    print("\n=== A. 位宽账的数学 ===")
    # 锚点:DSv4.1 发表的 2.609 bit。保留的是 int8,不是 bf16。
    dsv = MPS.avg_bits(2, 8, 39, 384)
    check("DSv4.1 hot39/384 保 int8 = 2.609 bit(已发表的锚点)",
          abs(dsv - 2.609375) < 1e-9, f"算出 {dsv}")
    ours16 = MPS.avg_bits(2, 16, 26, 256)
    check("同保留比例 hot26/256 保 bf16 = 3.422 bit",
          abs(ours16 - 3.421875) < 1e-9, f"算出 {ours16}")
    check("两者相差 0.81 bit —— 不是同一个工作点",
          abs((ours16 - dsv) - 0.8125) < 1e-9, f"差 {ours16 - dsv}")
    check("同保留比例 hot26/256 保 int8 回到 2.609",
          abs(MPS.avg_bits(2, 8, 26, 256) - 2.609375) < 1e-9)

    check("equiv_k(39, 384 -> 256) == 26", MPS.equiv_k(39, 384, 256) == 26,
          f"算出 {MPS.equiv_k(39, 384, 256)}")
    k_b = MPS.k_for_bits(dsv, 2, 16, 256)
    check("同位宽预算下 bf16 只能保 11 个", k_b == 11, f"算出 {k_b}")
    check("k_for_bits / avg_bits 往返一致(±半个专家)",
          abs(MPS.avg_bits(2, 16, k_b, 256) - dsv) < (16 - 2) * 0.5 / 256 + 1e-12)
    check("keep_bits == nbits 时 k_for_bits 返回 0 而不是除零",
          MPS.k_for_bits(2.0, 2, 2, 256) == 0)

    # 产物尺寸:工单 B.3 记的是 2bit 7.5 GiB
    g0 = MPS.product_gb(2, 16, 0, 256)
    check("K=0 时 2bit 产物 8.05 GB = 7.50 GiB(对上工单 §B.3)",
          abs(g0 - 8.0530) < 0.01 and abs(g0 / 2**30 * 1e9 - 7.5) < 0.02,
          f"{g0:.4f} GB = {g0 * 1e9 / 2**30:.3f} GiB")
    check("保 bf16 比保 int8 的产物大",
          MPS.product_gb(2, 16, 26, 256) > MPS.product_gb(2, 8, 26, 256))
    check("产物尺寸对 K 单调增",
          MPS.product_gb(2, 16, 0, 256) < MPS.product_gb(2, 16, 26, 256)
          < MPS.product_gb(2, 16, 52, 256))
    check("3bit 基座下 hot26 保 bf16 也比 3bit 裸版贵",
          MPS.avg_bits(3, 16, 26, 256) > 3.0)


# ---------------------------------------------------------------- B
def section_b() -> None:
    print("\n=== B. 选择逻辑 ===")
    # 手造计数:专家 3 最热,专家 0 最冷,1 和 2 并列
    cnt = {7: torch.tensor([0, 50, 50, 900], dtype=torch.long)}
    hot1 = MPS.select(cnt, 1, "hot")
    check("hot1 挑的是计数最大的那个", hot1 == {"7": [3]}, str(hot1))
    cold1 = MPS.select(cnt, 1, "cold")
    check("cold1 挑的是计数最小的那个", cold1 == {"7": [0]}, str(cold1))
    hot2 = MPS.select(cnt, 2, "hot")
    check("hot2 = {3, 并列里 id 小的那个}", hot2 == {"7": [1, 3]}, str(hot2))
    cold2 = MPS.select(cnt, 2, "cold")
    check("cold2 = {0, 并列里 id 大的那个}", cold2 == {"7": [0, 2]}, str(cold2))
    check("并列打破是 (-count, id),所以可复现",
          MPS.select(cnt, 2, "hot") == hot2)
    check("2K <= E 时 hot 和 cold 不相交",
          not (set(hot2["7"]) & set(cold2["7"])))
    check("K == E 时 hot 就是全部专家",
          MPS.select(cnt, 4, "hot") == {"7": [0, 1, 2, 3]})
    # 全零计数:排序没有意义,但不能崩
    z = {0: torch.zeros(4, dtype=torch.long)}
    check("全零计数下 select 不崩,且按 id 升序", MPS.select(z, 2, "hot") == {"0": [0, 1]})
    multi = MPS.select({0: torch.tensor([1, 2, 3, 4]), 5: torch.tensor([4, 3, 2, 1])}, 1, "hot")
    check("多层各自独立挑", multi == {"0": [3], "5": [0]}, str(multi))


# ---------------------------------------------------------------- C
def section_c(tmp: str) -> None:
    print("\n=== C. mp_select 端到端 ===")
    ck = os.path.join(tmp, "ckpt")
    acts = os.path.join(tmp, "acts")
    ckpt = make_ckpt(ck)
    make_const_acts(acts)

    r = run(SEL, "--bits-table", "--experts", "256", "--bits", "2")
    check("--bits-table 不读任何数据就能跑", r.returncode == 0, r.stderr[-400:])
    check("--bits-table 打出了 2.609 这个锚点", "2.609" in r.stdout, r.stdout[-300:])
    check("--bits-table 打出了 3.422(保 bf16 的代价)", "3.422" in r.stdout)

    r = run(SEL, "--in", ck, "--acts", acts, "--out", os.path.join(tmp, "k.json"))
    check("E=4 时默认 -k 算成 0,被守卫拒掉", r.returncode != 0, r.stdout[-300:])

    out = os.path.join(tmp, "keep.json")
    r = run(SEL, "--in", ck, "--acts", acts, "--out", out, "-k", str(TOPK), "--bits", "2")
    check("带 -k 正常跑完", r.returncode == 0, (r.stdout + r.stderr)[-600:])
    if r.returncode != 0:
        return

    with open(out, encoding="utf-8") as fh:
        hot = json.load(fh)
    check("keep json 是 {层号: [专家号]}",
          set(hot) == {str(ln) for ln in LAYERS}
          and all(isinstance(v, list) for v in hot.values()), str(hot))
    check("每层恰好 K 个专家", all(len(v) == TOPK for v in hot.values()), str(hot))
    check("专家号都在 [0, E)",
          all(0 <= e < E for v in hot.values() for e in v), str(hot))

    # 最强的那条:常量激活下路由是确定的,hot 必须精确等于被命中的专家
    for ln in LAYERS:
        want = routed_experts(ckpt, ln, acts)
        check(f"第 {ln} 层 hot{TOPK} 精确等于被路由命中的专家 {want}",
              hot[str(ln)] == want, f"算出 {hot[str(ln)]}")

    cold_path = os.path.join(tmp, "keep_cold.json")
    check("cold 变体也落盘了", os.path.isfile(cold_path))
    with open(cold_path, encoding="utf-8") as fh:
        cold = json.load(fh)
    check("cold 和 hot 不相交(E=4, K=2 刚好互补)",
          all(not (set(cold[k]) & set(hot[k])) for k in hot), f"hot={hot} cold={cold}")

    ipath = os.path.join(tmp, "keep_info.json")
    check("info 文件落盘了", os.path.isfile(ipath))
    with open(ipath, encoding="utf-8") as fh:
        info = json.load(fh)
    check("info 同时记了 bf16 和 int8 两个口径的平均位宽",
          abs(info["avg_bits_keep_bf16"] - MPS.avg_bits(2, 16, TOPK, E)) < 1e-6
          and abs(info["avg_bits_keep_int8"] - MPS.avg_bits(2, 8, TOPK, E)) < 1e-6,
          str({k: info[k] for k in ("avg_bits_keep_bf16", "avg_bits_keep_int8")}))
    # info 里的位宽是 round(..., 4),所以比的是"同样取整后相等",不是精确值。
    check("info 记了 DSv4.1 的参照位宽 2.6094",
          info["dsv41_reference_bits"] == round(MPS.avg_bits(2, 8, 39, 384), 4),
          f"存的是 {info['dsv41_reference_bits']},期望 {round(MPS.avg_bits(2, 8, 39, 384), 4)}")
    check("info 记了并列打破方式", info.get("tie_break") == "(-count, expert_id)")
    check("info 记了统计过哪些层和 checkpoint 里有哪些层",
          info["layers_counted"] == list(LAYERS) and info["layers_in_checkpoint"] == list(LAYERS))
    check("零 token 的专家被点出来了", "校准 token 为 0" in r.stdout or
          "非零专家 2/4" in r.stdout, r.stdout[-500:])

    for miss, flag in (("--in", ck), ("--acts", acts), ("--out", out)):
        argv = ["--in", ck, "--acts", acts, "--out", out, "-k", str(TOPK)]
        i = argv.index(miss)
        del argv[i:i + 2]
        rr = run(SEL, *argv)
        check(f"缺 {miss} 时报错退出", rr.returncode != 0, rr.stdout[-200:])


# ---------------------------------------------------------------- D
def section_d(tmp: str) -> None:
    print("\n=== D. offline_ptq --keep-hot 端到端 ===")
    ck = os.path.join(tmp, "ckpt")   # section_c 已经建好
    orig = read_dir(ck)
    keep_tbl = {str(ln): [0, 2] for ln in LAYERS}
    kp = os.path.join(tmp, "mixed.json")
    with open(kp, "w", encoding="utf-8") as fh:
        json.dump(keep_tbl, fh)

    dst = os.path.join(tmp, "out_mixed")
    r = run(PTQ_SCRIPT, "run", "--in", ck, "--out", dst, "--bits", "2",
            "--hessian", "identity", "--keep-hot", kp, "--full-manifest")
    check("--keep-hot 跑完,退出码 0", r.returncode == 0, (r.stdout + r.stderr)[-700:])
    if r.returncode != 0:
        return

    got = read_dir(dst)
    kept_bitexact = changed = 0
    for ln in LAYERS:
        for role in ("gate_up_proj", "down_proj"):
            k = f"model.language_model.layers.{ln}.mlp.experts.{role}"
            for e in range(E):
                same = torch.equal(orig[k][e], got[k][e])
                if e in keep_tbl[str(ln)]:
                    kept_bitexact += int(same)
                else:
                    changed += int(not same)
    n_kept_slots = len(LAYERS) * 2 * len(keep_tbl["0"])
    n_quant_slots = len(LAYERS) * 2 * (E - len(keep_tbl["0"]))
    check(f"保留的 {n_kept_slots} 个专家矩阵逐位未变",
          kept_bitexact == n_kept_slots, f"只有 {kept_bitexact} 个未变")
    check(f"没保留的 {n_quant_slots} 个专家矩阵都变了",
          changed == n_quant_slots, f"只有 {changed} 个变了")
    check("非专家张量也逐位未变",
          torch.equal(orig[f"model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"],
                      got[f"model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"]))

    with open(os.path.join(dst, "ptq_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    mp = man["mixed_precision"]
    want_bits = MPS.avg_bits(2, 16, 2, E)
    check("manifest 的平均位宽 = NB + (16-NB)*K/E",
          abs(mp["avg_bits"] - want_bits) < 1e-6, f"{mp['avg_bits']} vs {want_bits}")
    check("manifest 记了保留了多少个专家槽",
          mp["kept_experts"] == len(LAYERS) * 2 and mp["expert_slots"] == len(LAYERS) * E,
          str(mp))
    check("manifest 记了 keep_bits=16(保的是 bf16 不是 int8)", mp["keep_bits"] == 16)
    check("manifest 记了 keep json 的绝对路径", man["keep_hot"] == os.path.abspath(kp))
    check("manifest 每层的 kept 列表和输入一致",
          all(man["layers"][str(ln)]["kept"] == sorted(keep_tbl[str(ln)]) for ln in LAYERS))

    recs = man["layers"]["0"]["experts"]
    n_k = sum(1 for x in recs if x.get("kept"))
    n_s = sum(1 for x in recs if "w13_snr" in x)
    check("保留的专家不记 SNR(未量化的 SNR 是 +inf,会污染 p50)",
          n_k == 2 and n_s == E - 2 and n_k + n_s == E, f"kept={n_k} snr={n_s}")

    meta = shard_meta(dst)
    check("分片 metadata 带上了平均位宽",
          abs(float(meta.get("ptq_avg_bits", "nan")) - want_bits) < 1e-6, str(meta))
    check("分片 metadata 带上了保留专家数",
          meta.get("ptq_kept_experts") == str(len(LAYERS) * 2), str(meta))
    check("stdout 里把 int8 口径一起打了出来(防跨路线误比)",
          "int8" in r.stdout, r.stdout[-400:])

    # ---- 三类坏 keep json
    bads = [
        ("专家号越界", {str(ln): [0, E + 5] for ln in LAYERS}),
        ("保留了整层", {str(ln): list(range(E)) for ln in LAYERS}),
        ("键不是层号(把 info 文件传成了 keep)", {"tie_break": [0]}),
        ("值不是整数列表", {str(ln): ["0"] for ln in LAYERS}),
    ]
    for name, tbl in bads:
        p = os.path.join(tmp, "bad.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(tbl, fh)
        d2 = os.path.join(tmp, "out_bad")
        rr = run(PTQ_SCRIPT, "run", "--in", ck, "--out", d2, "--bits", "2",
                 "--hessian", "identity", "--keep-hot", p)
        check(f"坏 keep json 被拒:{name}", rr.returncode != 0,
              (rr.stdout + rr.stderr)[-300:])
        if os.path.isdir(d2):
            import shutil
            shutil.rmtree(d2)

    # ---- 缺层:警告但不拦
    p = os.path.join(tmp, "partial.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"0": [1]}, fh)
    d3 = os.path.join(tmp, "out_partial")
    rr = run(PTQ_SCRIPT, "run", "--in", ck, "--out", d3, "--bits", "2",
             "--hessian", "identity", "--keep-hot", p)
    check("keep json 缺层时照跑,但明确警告那层整层走低位宽",
          rr.returncode == 0 and "整层" in rr.stdout, (rr.stdout + rr.stderr)[-400:])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        section_a()
        section_b()
        section_c(tmp)
        section_d(tmp)
    print("\n" + "=" * 16 + f" {_ok} 项通过, {_fail} 项失败 " + "=" * 16)
    if _fail == 0:
        print("\n位宽账的要点:本路线保留的专家是 bf16,所以 hot26 是 3.42 bit,")
        print("不是 DSv4.1 那个 2.609。要和它的 65.66 比,用 hot11(2.609 bit)。")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
