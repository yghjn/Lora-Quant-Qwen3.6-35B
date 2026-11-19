#!/usr/bin/env python3
"""把"一遍语料是多少步"的估算,和上游**真的** ChatDataset 对拍。

    python verify/probe_steps_upstream_cpu.py --tokenizer <checkpoint 目录> --corpus <corpus.jsonl>

纯 CPU,不加载权重,不占卡;在训练环境里跑(要 torchtitan、datasets)。语料要先用
`prepare_corpus.py --seq-len 4096` 整理过(旁边有 `.meta.json`)。几千条的语料大约一两分钟。

`src/qwen36_steps.py` 算步数靠两个假设,这里各验一遍:

1. **每条样本在一行里占多少位置** = `prepare_corpus.py` 数出来的 token 数。
   判据:逐条和上游 `ChatDataset._tokenize_sample` 的 `input_ids` 长度比,必须全等。
2. **一遍要多少行**。上游先打乱(datasets 自己的随机数)、按卡切段、再贪心拼行;
   估算用的是另一个打乱顺序,哪张卡多哪张卡少对不上,但合计几乎不变。
   判据:把上游的数据集对每张卡真的迭代一遍数行数;"一遍的步数"(各卡平均)
   估算值与实测值相差不超过 max(1, 2%)。

退出码:0 通过;1 判据不成立;2 环境问题(缺包、文件不对)。
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import qwen36_steps as S  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--tokenizer", required=True, help="tokenizer 目录(tokenizer.json + chat template),即 checkpoint 目录")
ap.add_argument("--corpus", required=True, help="prepare_corpus.py --seq-len 的产物")
ap.add_argument("--seq-len", type=int, default=4096)
ap.add_argument("--world", type=int, default=S.DEFAULT_WORLD, help="卡数")
args = ap.parse_args()

try:
    meta = S.read_meta(args.corpus, args.seq_len)
except (S.StepsError, OSError) as exc:
    print(f"!! {exc}")
    sys.exit(2)

try:
    import torch  # noqa: F401  先于 datasets 导入,和训练进程的顺序一致
    from datasets import load_dataset
    from torchtitan.components.tokenizer import MultiModalTokenizer
    from torchtitan.hf_datasets.text_datasets import ChatDataset
    from torchtitan.models.qwen3_5 import QWEN3_5_SPECIAL_TOKENS
except Exception as exc:  # noqa: BLE001
    print(f"!! 环境不对({type(exc).__name__}: {exc})。在训练环境(lqq-train)里跑。")
    sys.exit(2)

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  {'OK ' if cond else 'X  '} {name}" + (f": {detail}" if detail else ""))
    ok = ok and bool(cond)


tok = MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS).build(tokenizer_path=args.tokenizer)
ds = load_dataset("json", data_files=args.corpus, split="train")
print(f"语料 {len(ds)} 条;记录里 {meta['samples']} 条 / {meta['tokens'] / 1e6:.2f} M token")


def _messages(sample):
    return sample["messages"]


print("\n== 1. 每条样本占的位置数")
one = ChatDataset(ds, tok, _messages, seq_len=args.seq_len)
real_n: list[int] = []
dropped = 0
for sample in ds:
    out = one._tokenize_sample(sample)
    if out is None:
        dropped += 1
        real_n.append(-1)
    else:
        real_n.append(len(out[0]))
check("条数一致", len(real_n) == len(meta["ntoks"]), f"上游 {len(real_n)} / 记录 {len(meta['ntoks'])}")
check("上游一条都没丢(超长的 prepare_corpus 已经先丢了)", dropped == 0, f"丢了 {dropped} 条")
diff = [(i, a, b) for i, (a, b) in enumerate(zip(real_n, meta["ntoks"])) if a != b]
hist: dict[int, int] = {}
for _, a, b in diff:
    hist[a - b] = hist.get(a - b, 0) + 1
check("逐条相等", not diff,
      f"{len(diff)} 条不等;(上游 - 记录) 的分布 {hist};前几条 (行号, 上游, 记录): {diff[:3]}" if diff else "")

print(f"\n== 2. 每张卡一遍要多少行({args.world} 张卡,seq_len {args.seq_len})")
real_rows = []
for rank in range(args.world):
    cd = ChatDataset(ds, tok, _messages, seq_len=args.seq_len, dp_rank=rank, dp_world_size=args.world, infinite=False)
    real_rows.append(sum(1 for _ in cd))
p = S.plan_from_ntoks(meta["ntoks"], args.seq_len, args.world, 1)
print(f"  上游实测: {real_rows}")
print(f"  估算:     {p['rows']}")
real = -(-sum(real_rows) // args.world)
est = p["per_epoch"]
tol = max(1, round(real * 0.02))
check("一遍的步数(各卡平均,向上取整):估算与实测相差不超过 max(1, 2%)", abs(real - est) <= tol,
      f"实测 {real},估算 {est},差 {est - real:+d}(允许 ±{tol})")
print(f"  各卡一遍的行数本来就不等:实测最少 {min(real_rows)}、最多 {max(real_rows)}。先到头的卡自己从头再来。")
epochs = S.epochs_from_env()
print(f"  {epochs:g} 遍:按实测是 {-(-sum(real_rows) * epochs // args.world):.0f} 步;"
      f"配置会用的是 {S.plan_from_ntoks(meta['ntoks'], args.seq_len, args.world, epochs)['steps']} 步")

print("\n结论:" + ("通过。" if ok else "未通过(退出码 1)。"))
sys.exit(0 if ok else 1)
