#!/usr/bin/env python3
"""用上游**真的** MoE 模块在 CPU 上走一次前向,验抓激活的钩子和离线对拍。需要训练环境。

    python3 verify/probe_capture_moe_cpu.py

`verify/test_capture_acts_cpu.py` 用的是照着真模型结构写的小模型 —— 它验不了"真模型
是不是真的长那样"。这个脚本把训练配置里第 0 层的 MoE 配置原样 build 出来(真的路由、
真的 token 分发、真的带 LoRA 的分组专家,256 个专家、真实尺寸,随机权重),套一个和
真模型同名的外壳(`layers.0.moe`),然后:

1. `qwen36_capture_acts.install` 能挂上:找到这一层、找到专家模块、找到 3 个 LoRA B;
2. 走一次前向,存下来的就是 MoE 的输入(去掉行尾补齐);
3. 对拍样本存下来了,而且用 `[gate; up]` 的顺序重算专家模块的输出,和模块自己算的
   对得上(反过来的顺序对不上)—— 这同时说明"打分乘在专家输出上"这个假设是对的;
4. `offline_ptq.route()` 用 router 权重重算的路由,和模块里实际用的一致。

纯 CPU,不占卡,约 4 GB 内存,一分钟以内。**没验到的**:8 卡并行下钩子拿到的张量
是不是同一种形状(那要上卡;抓激活时每步会核对形状,对不上会直接报错,不会存错的东西)。

退出码:0 通过;1 判据不成立;2 环境问题。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")
os.environ["HF_ASSETS_PATH"] = "/nonexistent-hf-assets-dir"
os.environ["DATA_FILES"] = "/nonexistent-corpus.jsonl"
os.environ["ADAPTER_SAVE_DIR"] = "/nonexistent-adapter-dir"
for _name in ("CAPTURE_ACTS_DIR", "CAPTURE_TOKENS", "CAPTURE_LAYERS", "INITIAL_LOAD_PATH"):
    os.environ.pop(_name, None)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok = ok and bool(cond)
    print(f"  {'OK ' if cond else 'X  '} {name}" + (f": {detail}" if detail else ""), flush=True)


def main() -> int:
    try:
        import torch

        import offline_ptq as ptq
        import qwen36_capture_acts as C
        import qwen36_lora_recipe as R
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"\n!! import 失败({type(exc).__name__}: {exc})—— 环境问题,先让 env/2_pip_train.sh 的闸门过了。")
        return 2

    torch.manual_seed(0)
    cfg = R.qwen36_35b_a3b_lora_smoke()
    moe_cfg = cfg.model_spec.model.layers[0].moe
    moe = moe_cfg.build()
    with torch.no_grad():
        for name, p in moe.named_parameters():
            if "lora_b" in name:
                p.zero_()
            else:
                p.normal_(std=0.05)
    inner = moe.routed_experts.inner_experts
    e_n, hidden, dim = inner.w1_EFD.shape
    top_k = moe.router.top_k
    print(f"真 MoE 已建: {type(moe).__name__} / {type(moe.routed_experts).__name__} / {type(inner).__name__},"
          f" E={e_n} dim={dim} hidden={hidden} top_k={top_k}")

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = moe

        def forward(self, h):
            return h + self.moe(h)

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = torch.nn.Embedding(100, dim)
            self.layers = torch.nn.ModuleDict({"0": Block()})

        def forward(self, tokens):
            x = self.tok_embeddings(tokens)
            for layer in self.layers.values():
                x = layer(x)
            return x

    root = Root()
    seq, pad = 96, 6
    tokens = torch.randint(1, 100, (1, seq))
    tokens[0, -pad - 1] = 1 if int(tokens[0, -pad - 2]) != 1 else 2
    tokens[0, -pad:] = 0

    seen: list = []
    moe.register_forward_pre_hook(lambda _m, a: seen.append(a[0].detach().clone()))

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "acts")
        try:
            n = C.install([root], steps=1, seq_len=seq, world=1, out_dir=out)
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            check("钩子能挂上真的 MoE 模块", False, f"{type(exc).__name__}: {exc}")
            return 1
        with open(os.path.join(out, C.META_NAME), encoding="utf-8") as fh:
            meta = json.load(fh)
        check("钩子能挂上真的 MoE 模块", n == 1 and meta["dim"] == dim and meta["lora_b_tensors_checked"] == 3,
              f"层数 {n},dim {meta['dim']},LoRA B 张量 {meta['lora_b_tensors_checked']} 个")

        try:
            with torch.no_grad():
                root(tokens)
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            check("真 MoE 在 CPU 上能前向", False, f"{type(exc).__name__}: {exc}")
            return 1
        check("真 MoE 在 CPU 上能前向", True)

        layers, _ = C.scan(out)
        parts = layers.get(0, [])
        good = len(parts) == 1 and parts[0][2] == seq - pad
        check("存了 1 个分片,token 数 = 去掉行尾补齐后的数", good, str([(r, s, k) for r, s, k, _ in parts]))
        if good:
            got = torch.load(parts[0][3], map_location="cpu", weights_only=True)
            want = seen[0].reshape(-1, dim)[: seq - pad].bfloat16()
            check("内容就是 MoE 的输入", got.dtype == torch.bfloat16 and torch.equal(got, want))
            check("offline_ptq.load_acts 读得回来", torch.equal(ptq.load_acts(out, 0, dim), got))

        ppath = os.path.join(out, C.PROBE_DIR, C.layer_dir(0) + ".pt")
        check("对拍样本存下来了(专家模块的调用方式和假设的一致)", os.path.isfile(ppath))
        if not os.path.isfile(ppath):
            return 1
        rec = torch.load(ppath, map_location="cpu", weights_only=True)
        # HF 文件里的 gate_up_proj 是 [E, 2F, D],前一半 gate(= w1)、后一半 up(= w3),
        # 见 qwen36_state_dict_adapter.py。这里按同样的方式拼出来。
        w13 = torch.cat([inner.w1_EFD.detach(), inner.w3_EFD.detach()], dim=1)
        w2 = inner.w2_EDF.detach()
        errs = C.probe_errors(rec["x"], rec["scores"], rec["ids"], rec["out"], w13, w2)
        check("按 [gate; up] 重算专家模块的输出,和模块自己算的对得上;反过来对不上",
              errs["gate_first"] < 0.05 and errs["up_first"] > 0.3,
              f"gate 在前 误差 {100 * errs['gate_first']:.2f}%,up 在前 误差 {100 * errs['up_first']:.0f}%")
        mine = ptq.route(rec["x"], moe.router.gate.weight.detach(), top_k)
        same = sum(set(mine[t].tolist()) == set(rec["ids"][t].tolist()) for t in range(mine.shape[0]))
        check("offline_ptq.route() 重算的路由和模块里实际用的一致", same >= 0.95 * mine.shape[0],
              f"{same}/{mine.shape[0]}")
        check("路由打分每个 token 加起来是 1(归一化过的 top-k)",
              bool(((rec["scores"].sum(-1) - 1).abs() < 1e-2).all()))

    print("\n结论:" + ("通过。" if ok else "未通过(退出码 1)。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
