#!/usr/bin/env python
"""让**未量化的基线模型自己**生成推理轨迹,当 LoRA 的训练语料(工单 §6.6)。

    python src/gen_trace.py --prompts <提问文件> --out traces.jsonl \\
        --url http://localhost:8900/v1/chat/completions --model <服务里的模型名> \\
        --n 1000 --max-tokens 3072

    python src/prepare_corpus.py --tokenizer <checkpoint 目录> --src traces.jsonl \\
        --seq-len 4096 --out corpus.jsonl          # 验模板、数 token、丢掉训练装不下的

只用标准库,哪个环境都能跑。要一个 OpenAI 兼容的 chat 接口(`vllm serve` 起的那种),
背后是**原始 bf16 权重**的 Qwen3.6。

这是什么、为什么
----------------
移植自 DSv4.1 的 `dsv4/unisvq/gen_trace.py`(`l84414662` 的工作区)。思路照搬:

    adapter 要补偿的是"量化之后的模型"和"原模型"之间的差。
    最直接的监督信号就是原模型自己的输出 —— 让量化 + adapter 的模型去复现它。

所以语料**必须是 Qwen3.6 自己生成的**,不能拿 DSv4.1 那批现成的轨迹:那是
DeepSeek 的输出,用它训等于教 Qwen 模仿另一个模型。能从那边复用的只有**提问**
(与模型无关)和下面几条实测出来的规矩。

从 DSv4.1 带过来的规矩
----------------------
* **截断的轨迹丢掉**(`finish_reason == "length"`)。它们在推理中途断掉,教的是
  "不收尾"。DSv4.1 第一批 600 条里有 111 条(18.5%)是这种,当时只计数没跳过。
  判据直接用服务返回的 `finish_reason` —— 按 token 数反推会漏掉"推理收尾了但
  答案被截断"的那些(他们实测漏了 39 条)。
* **没有推理、或者没有答案的丢掉。** 这批语料的用途就是推理分布 + 自己收尾。
* **在整个提问文件里均匀抽样**,不取前 N 条:提问文件往往是按领域拼起来的,
  顺序取会偏向排在前面的领域。
* 提问长度 40–6000 字符:太短没信息,太长会挤掉生成预算。

和 DSv4.1 版本不一样的地方
--------------------------
* **输出是 messages,不是拼好的文本。** DSv4.1 版自己拼
  `<｜User｜>…<｜Assistant｜><think>…</think>…`,而它踩过的最贵的坑正出在这里:
  角色标记写成了 ASCII 的 `<|User|>`,被拆成 5 个普通 token,修好后 LoRA 造成的
  退化从 +13.1 点压到 +3.5 点。这里**不碰模板**:生成时模板由服务端套,训练时由
  上游的 `ChatDataset` 套,两头用的都是 checkpoint 目录里那一份。
  每行::

      {"messages": [{"role": "user", "content": 提问},
                    {"role": "assistant", "content": 答案, "reasoning_content": 推理}],
       "finish_reason": "stop", "completion_tokens": N, "reasoning_tokens": N}

  Qwen3.6 的模板认 `reasoning_content`,会渲成 `<think>\\n推理\\n</think>\\n\\n答案`。
* **`--max-tokens` 要和训练的 seq_len 对齐。** 上游 `ChatDataset` 对超过 seq_len 的
  样本是**直接丢弃**,不截断。DSv4.1 那批推理 token 的中位是 2527、p75 是 5674,
  照搬 8192 的话,在我们 seq_len=4096 的配置下大半会被丢。默认 3072,给提问和模板
  留出余量;`prepare_corpus.py --seq-len` 会数出真正装得下的条数。
  想要更长的轨迹,得连训练的 seq_len 一起加(显存跟着涨)。
* 采样参数默认是评测那边用的(temperature 0.6 / top_p 0.95 / top_k 20),不是
  DSv4.1 的 1.0 / 0.95 —— 让语料的分布贴着部署时的分布。
* `--exclude`:把评测集的题目排除掉。**训练语料里混进评测题,复评的分数就不能信了。**
  可以直接给评测集的 parquet / arrow / json 文件或目录。判据是评测题**整题**出现在
  提问里(提问往往是"题目 + 选项",只做全文相等会漏)。
  注意 LiveCodeBench:DSv4.1 的提问池是对着 167 题的那个子集去污染的,它的来源里
  另有 888 道别的 LiveCodeBench 题。换用 1055 题的 release_v6 评测时**必须**重新排除。
* `--dump-prompts FILE`:只做筛选 / 去重 / 排除 / 抽样,把提问写出来就退出,不请求服务。
* 可以断点续跑(`--resume`),并发用线程池。
* 请求 localhost 时不走代理(环境里设了 http_proxy 的话,本机请求会被发到代理上去);
  其它地址照环境变量来。

提问文件的格式(自动识别,可以混着给)
------------------------------------
* `{"text": "<|User|>…<|Assistant|>…"}`  —— DSv4.1 的校准集 / 轨迹文件,ASCII 和全角
  标记都认,只取 User 段;
* `{"messages": [...]}`                  —— 取第一条 user;
* `{"prompt": …}` / `{"question": …}` / `{"instruction": …, "input": …}`。

`.jsonl` 每行一个对象,`.json` 是一个数组。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# DSv4.1 的文本格式:ASCII 和全角竖线都认(老语料是 ASCII 的)
_DS_USER_RE = re.compile(r"<[|｜]User[|｜]>(.*?)<[|｜]Assistant[|｜]>", re.DOTALL)
_THINK_CLOSE = "</think>"


def die(msg: str, *lines: str) -> None:
    print(f"\n!! {msg}", file=sys.stderr)
    for line in lines:
        print(f"   {line}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- 提问
def question_of(sample) -> str | None:
    """从一条样本里取出用户提问。认不出来返回 None。"""
    if isinstance(sample, str):
        return sample.strip() or None
    if not isinstance(sample, dict):
        return None
    text = sample.get("text")
    if isinstance(text, str):
        m = _DS_USER_RE.search(text)
        if m:
            return m.group(1).strip() or None
    msgs = sample.get("messages") or sample.get("conversations")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and (m.get("role") or m.get("from")) in ("user", "human"):
                content = m.get("content") if "content" in m else m.get("value")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None
    # problem / question_content 是评测集的字段名(MATH-500 / LiveCodeBench),给 --exclude 用
    for key in ("prompt", "question", "query", "problem", "question_content"):
        if isinstance(sample.get(key), str) and sample[key].strip():
            return sample[key].strip()
    if isinstance(sample.get("instruction"), str) and sample["instruction"].strip():
        q = sample["instruction"].strip()
        extra = sample.get("input")
        if isinstance(extra, str) and extra.strip():
            q = f"{q}\n\n{extra.strip()}"
        return q
    return None


_TABLE_COLUMNS = ("question", "problem", "question_content", "prompt", "text")


def _read_table(path: str):
    """读 .parquet / .arrow(评测集常见的落盘格式)。只取题目那几列,需要 pyarrow。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        die(f"读 {path} 需要 pyarrow。", "训练环境和评测环境里都有;或者先把题目导成 jsonl 再给。")
    if path.endswith(".parquet"):
        names = pq.read_schema(path).names
        cols = [c for c in _TABLE_COLUMNS if c in names]
        if not cols:
            die(f"{path} 里没有认得的题目列。", f"现有的列:{names};认得的:{list(_TABLE_COLUMNS)}")
        table = pq.read_table(path, columns=cols)
    else:
        try:
            table = pa.ipc.open_stream(path).read_all()
        except Exception:  # noqa: BLE001 - 不是流格式就按文件格式读
            table = pa.ipc.open_file(path).read_all()
        cols = [c for c in _TABLE_COLUMNS if c in table.column_names]
        if not cols:
            die(f"{path} 里没有认得的题目列。", f"现有的列:{table.column_names}")
        table = table.select(cols)
    yield from table.to_pylist()


def read_samples(path: str):
    """读 .jsonl(每行一个对象)、.json(一个数组)、.parquet / .arrow,或者一个目录
    (递归读里面这几种文件)。坏行跳过并计数。"""
    if os.path.isdir(path):
        found = []
        for root, _dirs, files in os.walk(path):
            found += [os.path.join(root, f) for f in files
                      if f.endswith((".jsonl", ".json", ".parquet", ".arrow"))]
        if not found:
            die(f"目录 {path} 里没有 jsonl / json / parquet / arrow 文件")
        for f in sorted(found):
            yield from read_samples(f)
        return
    if not os.path.isfile(path):
        die(f"找不到文件 {path}")
    if path.endswith((".parquet", ".arrow")):
        yield from _read_table(path)
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.read(1)
        fh.seek(0)
        if head == "[":
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                die(f"{path} 不是合法 JSON", str(exc))
            yield from data
            return
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


def norm(q: str) -> str:
    """比对用:去掉所有空白、统一大小写。只用来判"是不是同一道题"。"""
    return re.sub(r"\s+", "", q).lower()


def load_prompts(paths: list[str], min_chars: int, max_chars: int) -> tuple[list[str], dict[str, int]]:
    stats = {"读入": 0, "认不出": 0, "太短": 0, "太长": 0, "重复": 0}
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        for sample in read_samples(path):
            stats["读入"] += 1
            q = question_of(sample)
            if q is None:
                stats["认不出"] += 1
                continue
            if len(q) < min_chars:
                stats["太短"] += 1
                continue
            if len(q) > max_chars:
                stats["太长"] += 1
                continue
            key = norm(q)
            if key in seen:
                stats["重复"] += 1
                continue
            seen.add(key)
            out.append(q)
    return out, stats


_FINGERPRINT = 40  # 索引用:去空白、小写之后的前 40 个字符


def exclude_eval(prompts: list[str], eval_questions: list[str]) -> tuple[list[str], int, int]:
    """把"包含某道评测题"的提问剔掉。返回 (留下的提问, 剔掉几条, 涉及几道评测题)。

    判据:评测题(去空白、不分大小写)**整题**出现在提问里。提问往往是"题目 + 选项 /
    格式说明",只做全文相等会漏;只比开头又会把
    "Which of the following statements is incorrect?" 这种通用题干误判成撞题
    (2026-11-17 在 calib_v6 上实测:只比开头 40 字符报 25 条,核对整题后是 0 条)。
    所以先用开头 40 个字符做索引,命中后再核对整题。不足 40 字符的评测题只做全文相等。

    拦不住的:改写过的同一道题、只引用了题目一部分的提问。
    """
    index: dict[str, list[str]] = {}
    short: set[str] = set()
    for q in eval_questions:
        nq = norm(q)
        if len(nq) >= _FINGERPRINT:
            index.setdefault(nq[:_FINGERPRINT], []).append(nq)
        elif nq:
            short.add(nq)
    kept: list[str] = []
    hit_eval: set[str] = set()
    for p in prompts:
        np_ = norm(p)
        matched: list[str] = []
        if np_ in short:
            matched = [np_]
        else:
            for j in range(0, len(np_) - _FINGERPRINT + 1):
                cands = index.get(np_[j:j + _FINGERPRINT])
                if cands:
                    matched = [nq for nq in cands if np_.startswith(nq, j)]
                    if matched:
                        break
        if matched:
            hit_eval.update(matched)
        else:
            kept.append(p)
    return kept, len(prompts) - len(kept), len(hit_eval)


# ---------------------------------------------------------------- 请求
def _opener(url: str) -> urllib.request.OpenerDirector:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1") or host.startswith("127."):
        # 本机地址不走代理:环境里设了 http_proxy 时,urllib 会把 localhost 也发给代理。
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def split_reply(message: dict) -> tuple[str, str]:
    """(推理, 答案)。服务端开了 reasoning parser 时两者是分开的字段;没开时推理
    留在 content 里,形如 `推理…</think>\\n\\n答案`(生成提示已经以 `<think>\\n` 结尾)。"""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    if not reasoning and _THINK_CLOSE in content:
        head, _, tail = content.partition(_THINK_CLOSE)
        reasoning = head.replace("<think>", "", 1)
        content = tail
    return reasoning.strip(), content.strip()


def ask(opener, args, question: str) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k > 0:
        body["top_k"] = args.top_k
    req = urllib.request.Request(
        args.url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"}
    )
    with opener.open(req, timeout=args.timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", action="append", required=True, help="提问文件,可重复给")
    ap.add_argument("--out", help="输出 jsonl(生成时必填)")
    ap.add_argument("--url", default="http://localhost:8900/v1/chat/completions")
    ap.add_argument("--model", help="服务里的模型名(vllm serve 的 --served-model-name;生成时必填)")
    ap.add_argument("--dump-prompts", metavar="FILE",
                    help="只做到「筛选 + 去重 + 排除评测题 + 抽样」为止,把提问写成 jsonl 就退出,不请求服务。"
                         "用来先把一份干净的提问文件定下来")
    ap.add_argument("--n", type=int, default=1000, help="抽多少条提问(默认 1000;0 = 全部)")
    ap.add_argument("--seed", type=int, default=0, help="抽样的随机种子")
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="单条生成上限。要和训练 seq_len 对齐:超过 seq_len 的样本训练时直接丢弃")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20, help="<=0 则不传")
    ap.add_argument("--par", type=int, default=24, help="并发数,略低于服务的 --max-num-seqs")
    ap.add_argument("--timeout", type=float, default=1800.0, help="单条请求超时(秒)")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--exclude", action="append", default=[],
                    help="评测集的题目文件(格式同 --prompts),里面出现过的提问不用。可重复给")
    ap.add_argument("--keep-truncated", action="store_true",
                    help="保留 finish_reason=length 的轨迹(默认丢:它们教的是不收尾)")
    ap.add_argument("--resume", action="store_true", help="--out 已存在时接着跑,已有的提问不重问")
    args = ap.parse_args()
    if not args.dump_prompts and not (args.out and args.model):
        ap.error("生成时 --out 和 --model 都要给(只想导出提问文件就用 --dump-prompts)")

    prompts, stats = load_prompts(args.prompts, args.min_chars, args.max_chars)
    print("=== 提问 ===")
    print("  " + "  ".join(f"{k} {v}" for k, v in stats.items()) + f"  可用 {len(prompts)}")
    if not prompts:
        die("一条可用的提问都没有。", "支持的格式见本脚本开头的说明。")

    if args.exclude:
        for path in args.exclude:
            banned, ex_stats = load_prompts([path], 0, 10**9)
            if not banned:
                die(f"--exclude {path} 里一道题都没读出来。",
                    f"读入 {ex_stats['读入']},认不出 {ex_stats['认不出']}。宁可停下,也不要在没排除的情况下继续。")
            prompts, n_hit, n_eval = exclude_eval(prompts, banned)
            print(f"  排除评测题 {path}: {len(banned)} 道 -> 剔掉 {n_hit} 条提问(涉及 {n_eval} 道评测题),"
                  f"还剩 {len(prompts)}")
        print("  (判据是评测题整题出现在提问里;改写过的同一道题拦不住。)")
    else:
        print("  **没给 --exclude**:没有检查这些提问和评测集有没有重叠。")

    rng = random.Random(args.seed)
    if args.n > 0 and len(prompts) > args.n:
        prompts = rng.sample(prompts, args.n)  # 全文件均匀抽样,不取前 N 条
    else:
        rng.shuffle(prompts)

    if args.dump_prompts:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_prompts)) or ".", exist_ok=True)
        with open(args.dump_prompts, "w", encoding="utf-8") as fh:
            for q in prompts:
                fh.write(json.dumps({"prompt": q}, ensure_ascii=False) + "\n")
        lens = sorted(len(q) for q in prompts)
        n = len(lens)
        print(f"\n写出 {n} 条提问 -> {args.dump_prompts}")
        print(f"  长度(字符): min {lens[0]} p25 {lens[n // 4]} 中位 {lens[n // 2]} p75 {lens[3 * n // 4]} max {lens[-1]}")
        return 0

    done_keys: set[str] = set()
    if os.path.exists(args.out):
        if not args.resume:
            die(f"{args.out} 已存在。", "要接着跑加 --resume;要重来先把它挪走。")
        for sample in read_samples(args.out):
            q = question_of(sample)
            if q:
                done_keys.add(norm(q))
        prompts = [q for q in prompts if norm(q) not in done_keys]
        print(f"  续跑:已有 {len(done_keys)} 条,还要问 {len(prompts)} 条")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    print(f"\n=== 生成 ===\n  {args.url}  model={args.model}  max_tokens={args.max_tokens}  "
          f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}  并发 {args.par}")
    if not prompts:
        print("  没有要问的了。")
        return 0

    opener = _opener(args.url)
    lock = threading.Lock()
    counts = {"写出": 0, "截断": 0, "没有推理": 0, "没有答案": 0, "请求失败": 0}
    state = {"handled": 0}

    def tick() -> None:  # 调用方已持有 lock
        state["handled"] += 1
        if state["handled"] % 50 == 0:
            elapsed = time.time() - began
            eta = elapsed / state["handled"] * (len(prompts) - state["handled"])
            print(f"    已处理 {state['handled']}/{len(prompts)}  写出 {counts['写出']}  "
                  f"{elapsed:.0f}s  eta {eta:.0f}s", flush=True)
    rtoks: list[int] = []
    ctoks: list[int] = []
    errors: list[str] = []
    began = time.time()
    fh = open(args.out, "a", encoding="utf-8")

    def work(question: str) -> None:
        try:
            reply = ask(opener, args, question)
            choice = reply["choices"][0]
            reasoning, answer = split_reply(choice["message"])
            finish = choice.get("finish_reason")
            usage = reply.get("usage") or {}
            n_completion = int(usage.get("completion_tokens") or 0)
            n_reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        except Exception as exc:  # noqa: BLE001 - 网络/格式问题一律记一笔,不让一条拖垮整批
            with lock:
                counts["请求失败"] += 1
                if len(errors) < 3:
                    errors.append(repr(exc)[:300])
                tick()
            return
        with lock:
            tick()
            if finish == "length":
                counts["截断"] += 1
                if not args.keep_truncated:
                    return
            if not reasoning:
                counts["没有推理"] += 1
                return
            if not answer:
                counts["没有答案"] += 1
                return
            fh.write(json.dumps({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer, "reasoning_content": reasoning},
                ],
                "finish_reason": finish,
                "completion_tokens": n_completion,
                "reasoning_tokens": n_reasoning,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            counts["写出"] += 1
            rtoks.append(n_reasoning)
            ctoks.append(n_completion)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.par)) as pool:
            list(pool.map(work, prompts))
    finally:
        fh.close()

    print("\n=== 完成 ===")
    print("  " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    for err in errors:
        print(f"  !! 请求失败的例子: {err}")
    if counts["截断"]:
        kept = "保留了" if args.keep_truncated else "丢掉了"
        print(f"  截断率 {100 * counts['截断'] / len(prompts):.1f}%({kept})。偏高说明 --max-tokens 对这批提问太小,"
              "留下来的会偏向短轨迹。")

    def pct(values: list[int]) -> str:
        v = sorted(values)
        n = len(v)
        return f"min {v[0]} p25 {v[n // 4]} 中位 {v[n // 2]} p75 {v[3 * n // 4]} max {v[-1]}"

    if any(ctoks):
        print(f"  生成 token(推理+答案): {pct(ctoks)},合计 {sum(ctoks) / 1e6:.2f} M")
    if any(rtoks):
        print(f"  其中推理 token:        {pct(rtoks)}")
    print(f"  耗时 {(time.time() - began) / 60:.1f} 分钟 -> {args.out}")
    print("\n下一步(验模板、数 token、丢掉训练装不下的):")
    print(f"    python src/prepare_corpus.py --tokenizer <checkpoint 目录> --src {args.out} "
          "--seq-len 4096 --out corpus.jsonl")
    if counts["写出"] == 0 and not done_keys:
        die("一条都没写出来。", "看上面的计数和请求失败的例子。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
