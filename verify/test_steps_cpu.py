#!/usr/bin/env python3
"""src/qwen36_steps.py 的自测:正式训练的步数是怎么按语料量算出来的。

    python3 verify/test_steps_cpu.py

纯 CPU。A–F 只用标准库;G 把 prepare_corpus.py 真跑一遍,需要 jinja2 和 tokenizers
(训练环境里有;没有就跳过 G,不算失败)。

A. 拼行的行数,和一份**照抄上游生成器**的参照实现逐例对拍(随机 300 组)
B. 拼行的几个边界:正好塞满、差一个 token、超长样本、空语料
C. 按卡切段:和 datasets 的 contiguous shard 同一个公式
D. 步数 = 一遍的步数 × 遍数;语料条数少于卡数时拒绝
E. 语料旁边的 token 数记录:写、读、缺失 / 语料改过 / seq_len 不一致 / 损坏 都要拦住
F. 遍数的环境变量
G. prepare_corpus.py 带 --seq-len 时写出记录、不带时清掉旧记录

和上游**真的** ChatDataset 对拍(要真 tokenizer)在 verify/probe_steps_upstream_cpu.py。
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import qwen36_steps as S  # noqa: E402

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    print(f"  {'OK  ' if cond else 'X   '} {name}" + (f": {detail}" if detail else ""))
    if cond:
        passed += 1
    else:
        failed += 1


def upstream_rows(ntoks, seq_len: int) -> int:
    """上游 ChatDataset._iter_greedy_packed(086bf6c16)照抄:带缓冲区,数 yield 的次数。

    故意不和 S.pack_rows 共用任何代码 —— 那边是数数,这里是真的攒缓冲区。
    """
    rows = 0
    buf: list[int] = []
    for n in ntoks:
        if n > seq_len:  # _tokenize_sample 返回 None
            continue
        input_ids = [1] * n
        remaining = seq_len - len(buf)
        if len(input_ids) > remaining and len(buf) > 0:
            buf.extend([0] * remaining)
            assert len(buf) == seq_len
            rows += 1  # yield self._flush_buffers()
            buf = []
        buf.extend(input_ids)
        if len(buf) == seq_len:
            rows += 1
            buf = []
    if len(buf) > 0:
        rows += 1
    return rows


print("== A. 拼行的行数,和照抄上游的参照实现对拍")
rng = random.Random(7)
bad = []
for _ in range(300):
    seq = rng.choice([64, 256, 1000, 4096])
    xs = [rng.randint(1, int(seq * rng.choice([0.2, 0.6, 1.0, 1.3]))) for _ in range(rng.randint(0, 120))]
    if S.pack_rows(xs, seq) != upstream_rows(xs, seq):
        bad.append((seq, xs[:8]))
check("300 组随机语料,行数逐例相同", not bad, str(bad[:2]))

print("\n== B. 边界")
check("一条正好 4096 -> 1 行", S.pack_rows([4096], 4096) == 1)
check("4096 之后再来 1 个 token -> 2 行", S.pack_rows([4096, 1], 4096) == 2)
check("2000 + 2000 + 96 正好塞满 -> 1 行", S.pack_rows([2000, 2000, 96], 4096) == 1)
check("2000 + 2000 + 97 塞不下 -> 2 行", S.pack_rows([2000, 2000, 97], 4096) == 2)
check("超过 seq_len 的样本不占行(上游直接丢)", S.pack_rows([5000, 100, 9999], 4096) == 1)
check("空语料 -> 0 行", S.pack_rows([], 4096) == 0)
check("1000 token 一条,10 条 -> 3 行(4 + 4 + 2)", S.pack_rows([1000] * 10, 4096) == 3)

print("\n== C. 按卡切段")
parts = S.split_contiguous(10, 8)
check("10 条分 8 段:前 2 段各多一条", [len(p) for p in parts] == [2, 2, 1, 1, 1, 1, 1, 1], str([len(p) for p in parts]))
check("各段连续、不重不漏", [i for p in parts for i in p] == list(range(10)))
# datasets 的公式:start = div*i + min(i, mod); end = start + div + (1 if i < mod else 0)
ok = True
for n_items, world in ((2999, 8), (225, 8), (16, 8), (1000, 3)):
    div, mod = divmod(n_items, world)
    for i, p in enumerate(S.split_contiguous(n_items, world)):
        start = div * i + min(i, mod)
        ok = ok and (p.start, p.stop) == (start, start + div + (1 if i < mod else 0))
check("和 datasets 的 contiguous shard 公式一致", ok)

print("\n== D. 步数")
p1 = S.plan_from_ntoks([1000] * 80, 4096, world=8, epochs=1)
check("80 条 × 1000 token、8 卡:每卡 10 条 -> 一遍 3 步", p1["rows"] == [3] * 8 and p1["per_epoch"] == 3, str(p1["rows"]))
check("1 遍 -> 3 步", p1["steps"] == 3)
check("3 遍 -> 9 步", S.plan_from_ntoks([1000] * 80, 4096, 8, 3)["steps"] == 9)
check("2.5 遍 -> 向上取整 8 步", S.plan_from_ntoks([1000] * 80, 4096, 8, 2.5)["steps"] == 8)
check("0.1 遍也至少 1 步", S.plan_from_ntoks([1000] * 80, 4096, 8, 0.1)["steps"] == 1)
check("4 卡:每卡 20 条 -> 一遍 5 步", S.plan_from_ntoks([1000] * 80, 4096, 4, 1)["per_epoch"] == 5)
mixed = [random.Random(3).randint(200, 3000) for _ in range(2000)]
pa, pb = S.plan_from_ntoks(mixed, 4096, 8, 3), S.plan_from_ntoks(list(mixed), 4096, 8, 3)
check("同一份语料算两次,结果相同(8 个进程各算各的,必须一致)", pa == pb)
check("一遍的步数取各卡的平均(向上取整),不取最慢那张",
      pa["per_epoch"] == -(-sum(pa["rows"]) // 8), f"rows={pa['rows']}")
uneven = S.plan_from_ntoks([4000] * 9 + [100] * 71, 4096, 8, 1)
check("各卡行数不等时也是平均", uneven["per_epoch"] == -(-sum(uneven["rows"]) // 8) < max(uneven["rows"]),
      f"rows={uneven['rows']} -> {uneven['per_epoch']}")
lo = sum(mixed) / (8 * 4096)
check("一遍的步数不少于 token 合计 / (卡数 × seq_len)", pa["per_epoch"] >= lo, f"{pa['per_epoch']} >= {lo:.1f}")
check("装填率在 (0, 1]", 0 < pa["fill"] <= 1, f"{pa['fill']:.3f}")
try:
    S.plan_from_ntoks([1000] * 5, 4096, 8, 1)
    check("语料条数少于卡数时拒绝", False, "没抛")
except S.StepsError as exc:
    check("语料条数少于卡数时拒绝(分不到样本的卡会空转)", "空转" in str(exc), str(exc)[:40])
check("describe 里有条数、一遍的步数和总步数",
      all(s in S.describe(pa) for s in ("2000 条", f"一遍约 {pa['per_epoch']} 步", f"{pa['steps']} 步")), S.describe(pa))

print("\n== E. 语料旁边的 token 数记录")
with tempfile.TemporaryDirectory() as tmp:
    corpus = os.path.join(tmp, "corpus.jsonl")
    with open(corpus, "w", encoding="utf-8") as fh:
        for i in range(80):
            fh.write(json.dumps({"messages": [{"role": "user", "content": str(i)}]}) + "\n")
    mp = S.write_meta(corpus, [1000] * 80, 4096)
    check("记录写在 <语料>.meta.json", mp == corpus + ".meta.json" and os.path.isfile(mp))
    meta = S.read_meta(corpus, 4096)
    check("读回来:条数、token 合计、seq_len 都对",
          (meta["samples"], meta["tokens"], meta["seq_len"]) == (80, 80000, 4096))
    check("plan 读记录算出 9 步(默认 3 遍)", S.plan(corpus, 4096, 8, epochs=3)["steps"] == 9)

    def raises(fn, *want: str) -> tuple[bool, str]:
        try:
            fn()
            return False, "没抛"
        except S.StepsError as exc:
            return all(w in str(exc) for w in want), str(exc).replace("\n", " | ")[:70]

    good, msg = raises(lambda: S.read_meta(corpus, 2048), "seq_len=4096", "2048")
    check("seq_len 和训练配置不一致时拦住", good, msg)

    with open(corpus, "a", encoding="utf-8") as fh:
        fh.write("{}\n")
    good, msg = raises(lambda: S.read_meta(corpus, 4096), "改过", "prepare_corpus.py")
    check("语料在那之后改过(大小变了)时拦住,并说怎么重跑", good, msg)

    os.remove(mp)
    good, msg = raises(lambda: S.read_meta(corpus, 4096), "找不到", "prepare_corpus.py", "--training.steps")
    check("记录缺失时拦住,并给出两条出路", good, msg)

    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    good, msg = raises(lambda: S.read_meta(corpus, 4096), "读不了")
    check("记录损坏时拦住", good, msg)

    with open(mp, "w", encoding="utf-8") as fh:
        json.dump({"format": 99, "bytes": os.path.getsize(corpus), "seq_len": 4096, "ntoks": [1]}, fh)
    good, msg = raises(lambda: S.read_meta(corpus, 4096), "格式版本")
    check("格式版本不认识时拦住", good, msg)

print("\n== F. 遍数的环境变量")
check("没设 -> 默认 3", S.epochs_from_env({}) == 3.0 == S.DEFAULT_EPOCHS)
check("LORA_EPOCHS=1 -> 1", S.epochs_from_env({"LORA_EPOCHS": "1"}) == 1.0)
check("可以是小数", S.epochs_from_env({"LORA_EPOCHS": " 1.5 "}) == 1.5)
for raw in ("abc", "0", "-2", "nan", "inf"):
    try:
        S.epochs_from_env({"LORA_EPOCHS": raw})
        check(f"LORA_EPOCHS={raw} 被拒绝", False, "没抛")
    except S.StepsError:
        check(f"LORA_EPOCHS={raw} 被拒绝", True)

print("\n== G. prepare_corpus.py 写记录")
try:
    import jinja2  # noqa: F401
    from tokenizers import Tokenizer, models, pre_tokenizers

    have = True
except Exception as exc:  # noqa: BLE001
    have = False
    print(f"  跳过(缺 jinja2 或 tokenizers:{type(exc).__name__}: {exc})—— 训练环境里重跑")

if have:
    with tempfile.TemporaryDirectory() as tmp:
        tokdir = os.path.join(tmp, "tok")
        os.makedirs(tokdir)
        t = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        t.pre_tokenizer = pre_tokenizers.Whitespace()
        t.save(os.path.join(tokdir, "tokenizer.json"))
        template = (
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        with open(os.path.join(tokdir, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"chat_template": template}, fh)
        out = os.path.join(tmp, "c.jsonl")
        script = os.path.join(_SRC, "prepare_corpus.py")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.pop("LORA_EPOCHS", None)

        def run(*extra: str):
            return subprocess.run(
                [sys.executable, script, "--tokenizer", tokdir, "--synthetic", "40", "--out", out, *extra],
                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
            )

        r = run("--seq-len", "4096")
        check("带 --seq-len 跑通", r.returncode == 0, (r.stdout + r.stderr)[-200:] if r.returncode else "")
        mp = S.meta_path(out)
        if os.path.isfile(mp):
            meta = S.read_meta(out, 4096)
            with open(out, encoding="utf-8") as fh:
                n_lines = sum(1 for line in fh if line.strip())
            check("记录的条数 = 语料行数,大小对得上", meta["samples"] == n_lines == 40, f"{meta['samples']} / {n_lines}")
            check("每条的 token 数都是正数", all(n > 0 for n in meta["ntoks"]))
            p = S.plan(out, 4096, 8, epochs=3)
            check("输出里报了一遍的步数和总步数",
                  f"一遍约 {p['per_epoch']} 步" in r.stdout and f"-> {p['steps']} 步" in r.stdout, r.stdout[-260:])
            check("语料不到 2 M token 时有提醒", "低于建议的 2 M" in r.stdout)
        else:
            check("写出了 <语料>.meta.json", False, r.stdout[-200:])
        r = run()
        check("不带 --seq-len 重跑:旧记录被清掉,并说明正式配置会拒绝启动",
              r.returncode == 0 and not os.path.isfile(mp) and "拒绝启动" in r.stdout, r.stdout[-160:])

print(f"\n{passed} 通过, {failed} 失败")
sys.exit(1 if failed else 0)
