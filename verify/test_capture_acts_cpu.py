#!/usr/bin/env python3
"""src/qwen36_capture_acts.py 的 CPU 自测。只需要 torch + safetensors,不需要 torchtitan。

    python3 verify/test_capture_acts_cpu.py

用一个结构和真模型对得上的小模型(`layers.<N>.moe`、`moe.router.gate`、
`moe.routed_experts`,外面套一层模拟 activation checkpoint 的包装)走完整条路:

  A. 纯函数:行尾补齐怎么判、步数怎么算、层怎么选
  B. 钩子:存下来的就是每层 MoE 的输入,去掉了行尾补齐;反向重算不重复存;
     超过计划步数不再存;对拍样本和记录文件都在
  C. 该拦的都拦住:LoRA 的 B 不是零、目录里已有激活、磁盘不够、路径不对、
     权重目录是量化过的、激活里有 NaN
  D. `offline_ptq.py` 读得回来,而且和存进去的一样
  E. `check` 子命令:gate/up 顺序判得对;顺序反了能认出来;权重不是同一份时判不出来并报错;
     缺分片能发现

真模型上能不能跑通,这里验不了 —— 那要 8 张卡。这里验的是除此之外的全部。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import offline_ptq as ptq  # noqa: E402
import qwen36_capture_acts as C  # noqa: E402

for _name in (C.ENV_DIR, C.ENV_TOKENS, C.ENV_LAYERS):
    os.environ.pop(_name, None)

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


def raises(fn, want: str, exc_type=RuntimeError) -> tuple[bool, str]:
    try:
        fn()
        return False, "没抛异常"
    except exc_type as exc:
        return want in str(exc), str(exc)[:80]
    except BaseException as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 小模型
D, F, E, K, L, NL, V = 24, 8, 8, 2, 32, 3, 50
EOS = 0


class Inner(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.w13 = torch.nn.Parameter(torch.randn(E, 2 * F, D, generator=g) * 0.3)  # 前一半 gate
        self.w2 = torch.nn.Parameter(torch.randn(E, D, F, generator=g) * 0.3)
        self.w1_lora_b = torch.nn.Parameter(torch.zeros(E, F, 2))
        self.w13_lora_a = torch.nn.Parameter(torch.randn(E, 2, D, generator=g))


class Routed(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.inner_experts = Inner(g)

    def forward(self, x_BLD, topk_scores_BLK, topk_expert_ids_BLK, num_local_tokens_per_expert_E):
        ie = self.inner_experts
        out = torch.zeros_like(x_BLD)
        for b in range(x_BLD.shape[0]):
            for t in range(x_BLD.shape[1]):
                for k in range(K):
                    e = int(topk_expert_ids_BLK[b, t, k])
                    g_u = ie.w13[e] @ x_BLD[b, t]
                    a = torch.nn.functional.silu(g_u[:F]) * g_u[F:]
                    out[b, t] += topk_scores_BLK[b, t, k] * (ie.w2[e] @ a)
        return out


class Router(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.gate = torch.nn.Linear(D, E, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(torch.randn(E, D, generator=g))


class MoE(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.router = Router(g)
        self.routed_experts = Routed(g)

    def forward(self, x_BLD):
        scores = torch.softmax(self.router.gate(x_BLD), dim=-1)
        s, ids = torch.topk(scores, K, dim=-1)
        s = s / s.sum(-1, keepdim=True)
        return self.routed_experts(x_BLD, s, ids, None)


class Block(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.ffn_norm = torch.nn.LayerNorm(D)
        self.moe = MoE(g)

    def forward(self, x):
        return x + self.moe(self.ffn_norm(x))


class Wrapped(torch.nn.Module):
    """模拟 activation checkpoint 套的那一层:模块名里多一段。"""

    def __init__(self, inner):
        super().__init__()
        self._checkpoint_wrapped_module = inner

    def forward(self, x):
        return self._checkpoint_wrapped_module(x)


class Net(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.tok_embeddings = torch.nn.Embedding(V, D)
        with torch.no_grad():
            self.tok_embeddings.weight.copy_(torch.randn(V, D, generator=g))
        self.layers = torch.nn.ModuleDict({str(i): Wrapped(Block(g)) for i in range(NL)})

    def forward(self, tokens, **_kwargs):
        x = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            x = layer(x)
        return x

    def moe(self, i: int) -> MoE:
        return self.layers[str(i)]._checkpoint_wrapped_module.moe


def make_tokens(seed: int, pads: tuple[int, ...]) -> torch.Tensor:
    """每行末尾 pads[b] 个补齐。真数据用 1..V-1,保证行尾不会碰巧连着相同。"""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for p in pads:
        row = torch.randint(1, V, (L,), generator=g)
        row[-1] = 1 if int(row[-2]) != 1 else 2
        if p:
            row[L - p:] = EOS
        rows.append(row)
    return torch.stack(rows)


def spy_on(net: Net) -> dict[int, list[torch.Tensor]]:
    """自己另挂一套钩子记下每层 MoE 的输入,用来和被测的东西对。"""
    seen: dict[int, list[torch.Tensor]] = {i: [] for i in range(NL)}
    for i in range(NL):
        net.moe(i).register_forward_pre_hook(lambda _m, a, i=i: seen[i].append(a[0].detach().clone()))
    return seen


def write_ckpt(dirpath: str, net: Net, swap: bool = False) -> None:
    os.makedirs(dirpath, exist_ok=True)
    tensors = {}
    for i in range(NL):
        moe = net.moe(i)
        w13 = moe.routed_experts.inner_experts.w13.detach()
        if swap:
            w13 = torch.cat([w13[:, F:], w13[:, :F]], dim=1)
        p = f"model.language_model.layers.{i}.mlp."
        tensors[p + "experts.gate_up_proj"] = w13.contiguous()
        tensors[p + "experts.down_proj"] = moe.routed_experts.inner_experts.w2.detach().contiguous()
        tensors[p + "gate.weight"] = moe.router.gate.weight.detach().contiguous()
    save_file(tensors, os.path.join(dirpath, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(dirpath, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"text_config": {"num_experts": E, "num_experts_per_tok": K}}, fh)


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    # -X utf8:子进程的输出是中文,不依赖终端的 locale(-I 会忽略 PYTHONIOENCODING)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return subprocess.run([sys.executable, "-I", "-X", "utf8", *argv],
                          capture_output=True, text=True, encoding="utf-8", env=env)


# ---------------------------------------------------------------- A
def section_a() -> None:
    print("\n== A. 纯函数")
    t = make_tokens(1, (5, 0, 1))
    m = C.keep_mask(t)
    check("行尾 5 个补齐被去掉,前面的留着", m[0].tolist() == [True] * (L - 5) + [False] * 5)
    check("没有补齐的一行全留", bool(m[1].all()))
    check("只有 1 个补齐时分不出来,留着", bool(m[2].all()))
    check("整行都是同一个 token -> 全去掉", not bool(C.keep_mask(torch.zeros(1, L, dtype=torch.long)).any()))
    check("一维输入当成一行", tuple(C.keep_mask(t[0]).shape) == (1, L))
    check("100 万位置 / (8 卡 × 4096) -> 31 步", C.planned_steps(4096, 8, 1_000_000) == 31)
    check("要得再少也至少 1 步", C.planned_steps(4096, 8, 10) == 1)
    os.environ[C.ENV_TOKENS] = "2e5"
    check("CAPTURE_TOKENS=2e5 -> 7 步", C.planned_steps(4096, 8) == 7)
    os.environ[C.ENV_TOKENS] = "abc"
    good, msg = raises(lambda: C.planned_steps(4096, 8), "正数")
    check("CAPTURE_TOKENS 不是数字 -> 报错", good, msg)
    os.environ[C.ENV_TOKENS] = "0"
    good, msg = raises(lambda: C.planned_steps(4096, 8), "正数")
    check("CAPTURE_TOKENS=0 -> 报错", good, msg)
    os.environ.pop(C.ENV_TOKENS)
    check("磁盘上限: 31 步 × 8 卡 × 4096 × 2048 × 2 字节 × 40 层 ≈ 155 GiB",
          abs(C.bytes_needed(31, 8, 4096, 2048, 40) / 2**30 - 155.0) < 0.5,
          f"{C.bytes_needed(31, 8, 4096, 2048, 40) / 2**30:.1f}")
    check("层的写法: all / 单个 / 区间 / 列表",
          C.parse_layers(None, [0, 1, 2]) == [0, 1, 2] and C.parse_layers("1", [0, 1, 2]) == [1]
          and C.parse_layers("0-1", [0, 1, 2]) == [0, 1] and C.parse_layers("2,0", [0, 1, 2]) == [0, 2])
    good, msg = raises(lambda: C.parse_layers("7", [0, 1, 2]), "没有这些层")
    check("选了不存在的层 -> 报错", good, msg)
    check("分片目录名 offline_ptq 认得",
          ptq._ACT_DIR_RE.match(C.layer_dir(7)) is not None and int(ptq._ACT_DIR_RE.match(C.layer_dir(7)).group(1)) == 7
          and int(ptq._ACT_DIR_RE.match(C.layer_dir(20)).group(1)) == 20)


# ---------------------------------------------------------------- B
def section_b(tmp: str) -> tuple[Net, str, list[torch.Tensor]]:
    print("\n== B. 钩子")
    out = os.path.join(tmp, "acts")
    net = Net()
    spy = spy_on(net)
    n_layers = C.install([net], steps=2, seq_len=L, world=1, out_dir=out, source="/src", corpus="/corpus")
    check("3 层都挂上了(模块名里的包装段不碍事)", n_layers == NL)

    toks = [make_tokens(11, (5, 0)), make_tokens(12, (0, 9)), make_tokens(13, (0, 0))]
    with torch.no_grad():
        net(toks[0])
        # 模拟 activation checkpoint 在反向时把第 1 层重算一遍
        net.layers["1"](torch.randn(2, L, D))
        net(toks[1])
        net(toks[2])  # 第 3 次前向,超过计划的 2 步

    layers, leftovers = C.scan(out)
    check("每层恰好 2 个分片(重算没多存,第 3 步没存)",
          sorted(layers) == [0, 1, 2] and all(len(v) == 2 for v in layers.values()),
          str({k: len(v) for k, v in layers.items()}))
    check("没有残留的 .tmp", not leftovers)
    want_n = [2 * L - 5, 2 * L - 9]
    check("文件名里的 token 数 = 去掉补齐后的数",
          all([n for _, _, n, _ in layers[i]] == want_n for i in range(NL)), str(layers[0]))

    same = True
    dtype_ok = True
    for i in range(NL):
        for step, (_, _, n, path) in enumerate(layers[i]):
            got = torch.load(path, map_location="cpu", weights_only=True)
            dtype_ok &= got.dtype == torch.bfloat16 and tuple(got.shape) == (n, D)
            # spy 的记录里第 1 层多一次(那次"重算"),按前向次序取:0->第 0 条,1->最后第 2 条
            ref_all = [x for x in spy[i] if tuple(x.shape) == (2, L, D)]
            ref = ref_all[0] if step == 0 else ref_all[-2]
            keep = C.keep_mask(toks[step]).reshape(-1)
            same &= torch.equal(got, ref.reshape(-1, D)[keep].bfloat16())
    check("存下来的是 bf16 的 [n, dim]", dtype_ok)
    check("内容就是每层 MoE 的输入(去掉补齐的位置之后)", same)

    probes = sorted(os.listdir(os.path.join(out, C.PROBE_DIR)))
    check("每层一份对拍样本", probes == [C.layer_dir(i) + ".pt" for i in range(NL)], str(probes))
    rec = torch.load(os.path.join(out, C.PROBE_DIR, probes[0]), map_location="cpu", weights_only=True)
    check("对拍样本里有输入 / 打分 / 专家号 / 输出,形状对",
          tuple(rec["x"].shape) == (L, D) and tuple(rec["scores"].shape) == (L, K)
          and tuple(rec["ids"].shape) == (L, K) and tuple(rec["out"].shape) == (L, D))
    with open(os.path.join(out, C.META_NAME), encoding="utf-8") as fh:
        meta = json.load(fh)
    check("记录文件里有来源、步数、层",
          meta["source"] == "/src" and meta["steps"] == 2 and meta["layers"] == [0, 1, 2] and meta["dim"] == D
          and meta["lora_b_tensors_checked"] == NL, str(meta))
    return net, out, toks


# ---------------------------------------------------------------- C
def section_c(tmp: str, first_out: str) -> None:
    print("\n== C. 该拦的都拦住")
    good, msg = raises(lambda: C.install([Net()], steps=1, seq_len=L, world=1, out_dir=first_out), "已经有激活")
    check("目录里已有激活 -> 拒绝(两趟混在一起分不出来)", good, msg)
    good, msg = raises(lambda: C.install([Net()], steps=1, seq_len=L, world=1), C.ENV_DIR)
    check("没给输出目录 -> 拒绝", good, msg)
    good, msg = raises(lambda: C.install([Net()], steps=1, seq_len=L, world=1, out_dir="acts"), "绝对路径")
    check("输出目录是相对路径 -> 拒绝", good, msg)

    net = Net()
    out = os.path.join(tmp, "acts_lora")
    C.install([net], steps=3, seq_len=L, world=1, out_dir=out)
    with torch.no_grad():
        net(make_tokens(21, (0, 0)))
        net.moe(1).routed_experts.inner_experts.w1_lora_b[0, 0, 0] = 1e-3
    good, msg = raises(lambda: net(make_tokens(22, (0, 0))), "不是全零")
    check("LoRA 的 B 不再是零 -> 下一步前向立刻报错", good, msg)

    real = C.shutil.disk_usage
    try:
        C.shutil.disk_usage = lambda _p: type("U", (), {"free": 1000})()
        good, msg = raises(
            lambda: C.install([Net()], steps=1, seq_len=L, world=1, out_dir=os.path.join(tmp, "acts_full")),
            C.ENV_TOKENS)
        check("磁盘不够 -> 开跑之前就报,并说调哪个变量", good, msg)
    finally:
        C.shutil.disk_usage = real

    net = Net()
    out = os.path.join(tmp, "acts_one")
    check("CAPTURE_LAYERS 只选第 1 层 -> 只挂 1 层",
          C.install([net], steps=1, seq_len=L, world=1, out_dir=out, layers_spec="1") == 1)
    with torch.no_grad():
        net(make_tokens(23, (0, 3)))
    layers, _ = C.scan(out)
    check("只有第 1 层的目录,对拍样本也只有它",
          sorted(layers) == [1] and os.listdir(os.path.join(out, C.PROBE_DIR)) == [C.layer_dir(1) + ".pt"])

    net = Net()
    C.install([net], steps=1, seq_len=L, world=1, out_dir=os.path.join(tmp, "acts_nan"))
    with torch.no_grad():
        net.tok_embeddings.weight[3, 0] = float("nan")
    tok = make_tokens(24, (0, 0))
    tok[0, 0] = 3
    good, msg = raises(lambda: net(tok), "NaN")
    check("激活里有 NaN -> 报错,不存", good, msg)

    net = Net()
    C.install([net], steps=1, seq_len=L, world=1, out_dir=os.path.join(tmp, "acts_float"))
    good, msg = raises(lambda: net(torch.randn(2, L)), "token id")
    check("模型输入不是 token id -> 报错(行尾补齐没法判)", good, msg)

    for name in C.QUANTIZED_MARKERS:
        d = os.path.join(tmp, "q_" + name)
        os.makedirs(d)
        open(os.path.join(d, name), "w").close()
        good, msg = raises(lambda d=d: C.refuse_quantized_source(d), "原始权重")
        check(f"权重目录里有 {name} -> 拒绝", good, msg)
    C.refuse_quantized_source(tmp)
    check("普通目录不拦", True)


# ---------------------------------------------------------------- D
def section_d(tmp: str, net: Net, out: str) -> str:
    print("\n== D. offline_ptq 读得回来")
    layers, _ = C.scan(out)
    ok = True
    for i in range(NL):
        want = torch.cat([torch.load(p, map_location="cpu", weights_only=True) for _, _, _, p in layers[i]])
        got = ptq.load_acts(out, i, D)
        ok &= torch.equal(got, want)
    n_tok = sum(n for _, _, n, _ in layers[0])
    check("load_acts 读分片目录,按顺序拼,和存进去的一样", ok)
    check("token 数对", ptq.load_acts(out, 0, D).shape[0] == n_tok == 4 * L - 14)

    ckpt = os.path.join(tmp, "ckpt")
    write_ckpt(ckpt, net)
    hess = os.path.join(tmp, "hess")
    r = run_cli(os.path.join(ROOT, "src", "offline_ptq.py"), "hessian", "--in", ckpt, "--acts", out, "--out", hess)
    good = r.returncode == 0 and os.path.isfile(os.path.join(hess, "hessian_layer00.pt"))
    check("offline_ptq.py hessian 直接吃这个目录", good, (r.stdout + r.stderr)[-300:])
    if good:
        h = torch.load(os.path.join(hess, "hessian_layer00.pt"), map_location="cpu", weights_only=True)
        check("每个 token 进 top_k 个专家:命中数合计 = token 数 × top_k",
              int(h["ntok"].sum()) == n_tok * K, f"{int(h['ntok'].sum())} vs {n_tok * K}")

    both = os.path.join(tmp, "acts_both")
    os.makedirs(os.path.join(both, "layer00"))
    torch.save(torch.zeros(4, D), os.path.join(both, "layer00", "a.pt"))
    torch.save(torch.zeros(4, D), os.path.join(both, "layer0.pt"))
    good, msg = raises(lambda: ptq.load_acts(both, 0, D), "", SystemExit)
    check("同一层既有文件又有目录 -> 拒绝", good, msg)
    os.remove(os.path.join(both, "layer0.pt"))
    open(os.path.join(both, "layer00", "b.pt.tmp"), "w").close()
    good, msg = raises(lambda: ptq.load_acts(both, 0, D), "", SystemExit)
    check("分片目录里有没写完的 .tmp -> 拒绝", good, msg)
    os.remove(os.path.join(both, "layer00", "b.pt.tmp"))
    torch.save(torch.zeros(4, D + 1), os.path.join(both, "layer00", "c.pt"))
    good, msg = raises(lambda: ptq.load_acts(both, 0, D), "", SystemExit)
    check("某个分片的 dim 不对 -> 拒绝", good, msg)
    single = os.path.join(tmp, "acts_single")
    os.makedirs(single)
    torch.save(torch.ones(5, D), os.path.join(single, "layer0.pt"))
    check("原来的单文件放法还认", tuple(ptq.load_acts(single, 0, D).shape) == (5, D))
    return ckpt


# ---------------------------------------------------------------- E
def section_e(tmp: str, net: Net, out: str, ckpt: str) -> None:
    print("\n== E. check 子命令")
    tool = os.path.join(ROOT, "src", "qwen36_capture_acts.py")

    with torch.no_grad():
        rec = torch.load(os.path.join(out, C.PROBE_DIR, C.layer_dir(0) + ".pt"), map_location="cpu", weights_only=True)
        ie = net.moe(0).routed_experts.inner_experts
        errs = C.probe_errors(rec["x"], rec["scores"], rec["ids"], rec["out"], ie.w13.detach(), ie.w2.detach())
    check("重算:对的顺序误差接近 0,反的顺序误差很大",
          errs["gate_first"] < 1e-4 and errs["up_first"] > 0.3, str(errs))

    r = run_cli(tool, "check", "--acts", out)
    check("不给 --in:只核对文件,通过", r.returncode == 0 and "只核对了文件" in r.stdout, (r.stdout + r.stderr)[-300:])

    r = run_cli(tool, "check", "--acts", out, "--in", ckpt)
    check("给了对的权重:通过,判出 gate_first,路由全一致",
          r.returncode == 0 and "--gate-up-order gate_first" in r.stdout and f"路由一致 {L}/{L}" in r.stdout
          and "每专家平均" in r.stdout, (r.stdout + r.stderr)[-500:])

    swapped = os.path.join(tmp, "ckpt_swapped")
    write_ckpt(swapped, net, swap=True)
    r = run_cli(tool, "check", "--acts", out, "--in", swapped)
    check("权重文件里 gate/up 反着放:判出 up_first,并提醒必须显式传",
          r.returncode == 0 and "--gate-up-order up_first" in r.stdout and "必须显式传" in r.stdout,
          (r.stdout + r.stderr)[-500:])

    other = os.path.join(tmp, "ckpt_other")
    write_ckpt(other, Net(seed=99))
    r = run_cli(tool, "check", "--acts", out, "--in", other)
    check("权重不是抓激活时那一份:判不出来,退出码 1",
          r.returncode == 1 and "判不出来" in r.stdout, (r.stdout + r.stderr)[-500:])

    layers, _ = C.scan(out)
    victim = layers[2][1][3]
    os.rename(victim, victim + ".bak")
    r = run_cli(tool, "check", "--acts", out)
    check("某层少一个分片:发现,退出码 1", r.returncode == 1 and "不一致" in r.stdout, (r.stdout + r.stderr)[-300:])
    os.rename(victim + ".bak", victim)
    open(victim + ".tmp", "w").close()
    r = run_cli(tool, "check", "--acts", out)
    check("有没写完的 .tmp:发现,退出码 1", r.returncode == 1 and ".tmp" in r.stdout, (r.stdout + r.stderr)[-300:])
    os.remove(victim + ".tmp")

    with open(os.path.join(out, C.META_NAME), encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["steps"] = 5
    with open(os.path.join(out, C.META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    r = run_cli(tool, "check", "--acts", out)
    check("分片数比计划的少(没抓完):发现,退出码 1", r.returncode == 1 and "没抓完" in r.stdout, (r.stdout + r.stderr)[-300:])

    r = run_cli(tool, "check", "--acts", os.path.join(tmp, "nope"))
    check("目录不存在:退出码 2", r.returncode == 2)


def main() -> int:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        section_a()
        net, out, _ = section_b(tmp)
        section_c(tmp, out)
        ckpt = section_d(tmp, net, out)
        section_e(tmp, net, out, ckpt)
    print(f"\n{passed} 通过, {failed} 失败")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
