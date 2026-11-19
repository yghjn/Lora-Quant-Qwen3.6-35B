"""正式训练该跑多少步:按语料量算,不写死(2026-11-19)。

只用标准库,``prepare_corpus.py`` 和 ``qwen36_lora_recipe.py`` 共用。

为什么是按 token 算而不是按条数算
--------------------------------
上游 ``ChatDataset``(``torchtitan/hf_datasets/text_datasets.py`` 的
``_iter_greedy_packed``,086bf6c16)**不是一条样本占一行**。它把样本一条接一条往
一行里塞,塞到 ``seq_len`` 满、或者下一条塞不下了才换行(剩下的位置补 EOS、不算 loss)。
所以一步吃掉的是 ``卡数 × seq_len`` 个 token 位置,不是 ``卡数`` 条样本:
平均 1000 token 一条的语料,一行里大约拼 4 条。

"一遍语料 = 多少步"因此要按每条的 token 数把拼行过程走一遍才知道:

1. ``prepare_corpus.py --seq-len N`` 数出每条样本在一行里占几个位置,写进
   ``<语料>.meta.json``(本模块的 ``write_meta``);
2. 训练配置读它(``plan``),按上游同样的办法 —— 打乱、按卡切成连续的几段、
   每段贪心拼行 —— 算出每张卡一遍要多少行,**各卡取平均**当"一遍的步数";
3. 步数 = 一遍的步数 × 遍数(环境变量 ``LORA_EPOCHS``,默认 ``DEFAULT_EPOCHS``)。

"一遍"是对整体说的,不是对每张卡:各卡分到的 token 数不一样,一遍要的行数也不一样
(2026-11-19 拿 2848 条 / 4.09 M token 的语料在上游真的 ChatDataset 上数过:
各卡 152~176 行,平均 159.4;这里估出来 159.5)。先到头的卡自己从头再来,
所以跑完"一遍的步数"时,有的卡多看了几条,有的卡还差几条。

为什么取平均而不取最慢的那张卡:上游打乱用的是 datasets 自己的随机数,这里的顺序
和它不同,哪张卡多哪张卡少对不上(同一份语料,上游最多的是 176,这里最多的是 164);
合计则几乎不随顺序变。对拍脚本:``verify/probe_steps_upstream_cpu.py``。
"""

from __future__ import annotations

import json
import math
import os
import random

META_SUFFIX = ".meta.json"
META_FORMAT = 1
DEFAULT_EPOCHS = 3.0
DEFAULT_WORLD = 8
SHUFFLE_SEED = 42
EPOCHS_ENV = "LORA_EPOCHS"


class StepsError(ValueError):
    """算不出步数。消息是写给人看的,直接打印即可。"""


def pack_rows(ntoks, seq_len: int) -> int:
    """按上游的贪心拼行,这些样本要占多少行。

    上游逻辑(``_iter_greedy_packed``)::

        塞不下且这一行不是空的 -> 这一行收尾,换行
        塞进去
        正好塞满 -> 换行
        数据到头,这一行不是空的 -> 收尾

    超过 ``seq_len`` 的样本上游直接丢,这里也跳过。
    """
    rows = 0
    used = 0
    for n in ntoks:
        n = int(n)
        if n <= 0 or n > seq_len:
            continue
        if n > seq_len - used and used > 0:
            rows += 1
            used = 0
        used += n
        if used == seq_len:
            rows += 1
            used = 0
    if used > 0:
        rows += 1
    return rows


def split_contiguous(n_items: int, world: int) -> list[range]:
    """datasets 的 ``split_dataset_by_node`` 对普通数据集的切法:连续的 ``world`` 段。

    即 ``Dataset.shard(num_shards=world, index=rank, contiguous=True)``:
    前 ``n_items % world`` 段各多一条。
    """
    div, mod = divmod(n_items, world)
    out = []
    start = 0
    for i in range(world):
        size = div + (1 if i < mod else 0)
        out.append(range(start, start + size))
        start += size
    return out


def rows_per_rank(ntoks, seq_len: int, world: int, seed: int = SHUFFLE_SEED) -> list[int]:
    """每张卡把自己那一段语料过一遍要多少行(= 多少步)。"""
    order = [int(n) for n in ntoks]
    random.Random(seed).shuffle(order)
    return [pack_rows([order[i] for i in part], seq_len) for part in split_contiguous(len(order), world)]


def meta_path(corpus_path: str) -> str:
    return corpus_path + META_SUFFIX


def write_meta(corpus_path: str, ntoks, seq_len: int) -> str:
    """把每条样本的 token 数记在语料旁边。``ntoks`` 要和语料的行一一对应。"""
    ntoks = [int(n) for n in ntoks]
    meta = {
        "format": META_FORMAT,
        "corpus": os.path.basename(corpus_path),
        "bytes": os.path.getsize(corpus_path),
        "samples": len(ntoks),
        "seq_len": int(seq_len),
        "tokens": sum(ntoks),
        "ntoks": ntoks,
    }
    path = meta_path(corpus_path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
        fh.write("\n")
    return path


def _redo(corpus_path: str, seq_len: int) -> str:
    return (
        f"重跑一次(几秒钟): python src/prepare_corpus.py --tokenizer <checkpoint 目录> "
        f"--src <生成的 traces.jsonl> --seq-len {seq_len} --out {corpus_path}\n"
        f"或者自己定步数: 训练命令末尾加 --training.steps N"
    )


def read_meta(corpus_path: str, seq_len: int) -> dict:
    """读语料旁边的 token 数记录,并确认它说的就是这份语料。对不上就抛 ``StepsError``。"""
    path = meta_path(corpus_path)
    if not os.path.isfile(path):
        raise StepsError(
            f"找不到 {path} —— 没有每条样本的 token 数,算不出正式训练该跑多少步。\n"
            f"它由 prepare_corpus.py 在带 --seq-len 时写出。\n" + _redo(corpus_path, seq_len)
        )
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        ntoks = [int(n) for n in meta["ntoks"]]
        fmt, size, meta_seq = int(meta["format"]), int(meta["bytes"]), int(meta["seq_len"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StepsError(f"{path} 读不了({type(exc).__name__}: {exc})。\n" + _redo(corpus_path, seq_len)) from None
    if fmt != META_FORMAT:
        raise StepsError(f"{path} 的格式版本是 {fmt},这份代码认的是 {META_FORMAT}。\n" + _redo(corpus_path, seq_len))
    actual = os.path.getsize(corpus_path)
    if size != actual:
        raise StepsError(
            f"{path} 记的语料大小是 {size} 字节,而 {corpus_path} 现在是 {actual} 字节 —— "
            f"语料在那之后改过了,记录作废。\n" + _redo(corpus_path, seq_len)
        )
    if meta_seq != int(seq_len):
        raise StepsError(
            f"{path} 是按 seq_len={meta_seq} 数的,训练配置的 seq_len 是 {seq_len}。\n" + _redo(corpus_path, seq_len)
        )
    if not ntoks:
        raise StepsError(f"{path} 里一条样本都没有。")
    meta["ntoks"] = ntoks
    return meta


def epochs_from_env(env=None) -> float:
    """遍数:环境变量 ``LORA_EPOCHS``,没设就是 ``DEFAULT_EPOCHS``。可以是小数。"""
    raw = ((env if env is not None else os.environ).get(EPOCHS_ENV) or "").strip()
    if not raw:
        return DEFAULT_EPOCHS
    try:
        value = float(raw)
    except ValueError:
        raise StepsError(f"{EPOCHS_ENV}={raw!r} 不是数字。") from None
    if not (value > 0 and math.isfinite(value)):
        raise StepsError(f"{EPOCHS_ENV}={raw!r} 必须大于 0。")
    return value


def plan_from_ntoks(ntoks, seq_len: int, world: int = DEFAULT_WORLD, epochs: float = DEFAULT_EPOCHS) -> dict:
    """由每条的 token 数算步数。返回一张说明白了的表(见下面的键)。"""
    ntoks = [int(n) for n in ntoks if 0 < int(n) <= seq_len]
    if world < 1:
        raise StepsError(f"卡数 {world} 不对。")
    if len(ntoks) < world:
        raise StepsError(
            f"语料只有 {len(ntoks)} 条,不够 {world} 张卡一卡一条 —— 分不到样本的卡会一直空转。"
        )
    rows = rows_per_rank(ntoks, seq_len, world)
    # 各卡的平均,不取最慢那张:哪张卡分到的 token 多是打乱顺序决定的,最大值跟着顺序跳
    # (实测 152~176 行),而合计几乎不随顺序变(实测 1275,估算 1276)。
    per_epoch = math.ceil(sum(rows) / world)
    tokens = sum(ntoks)
    return {
        "samples": len(ntoks),
        "tokens": tokens,
        "seq_len": int(seq_len),
        "world": int(world),
        "rows": rows,
        "per_epoch": per_epoch,
        "epochs": float(epochs),
        "steps": max(1, math.ceil(per_epoch * float(epochs) - 1e-9)),
        # 一行里真样本占的比例(其余是换行时补的 EOS,不算 loss)
        "fill": tokens / float(sum(rows) * seq_len),
        "per_row": len(ntoks) / float(sum(rows)),
    }


def plan(corpus_path: str, seq_len: int, world: int = DEFAULT_WORLD, epochs: float | None = None) -> dict:
    """读语料旁边的记录,算正式训练的步数。"""
    meta = read_meta(corpus_path, seq_len)
    return plan_from_ntoks(
        meta["ntoks"], seq_len, world, epochs_from_env() if epochs is None else epochs
    )


def describe(p: dict) -> str:
    """一行话说清步数是怎么来的。"""
    return (
        f"语料 {p['samples']} 条 / {p['tokens'] / 1e6:.2f} M token;"
        f"{p['world']} 张卡、每行拼满 {p['seq_len']} token(平均一行 {p['per_row']:.1f} 条,"
        f"装填率 {p['fill'] * 100:.0f}%)-> 一遍约 {p['per_epoch']} 步;"
        f"{p['epochs']:g} 遍 -> {p['steps']} 步"
    )
