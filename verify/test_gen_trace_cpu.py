#!/usr/bin/env python3
"""src/gen_trace.py 的端到端自测,外加它的产物过 src/prepare_corpus.py。

    python3 verify/test_gen_trace_cpu.py

纯 CPU、不需要模型:在本机起一个假的 OpenAI 兼容服务,按提问里的记号返回各种情况
(推理在 reasoning_content / reasoning 字段里、留在 content 里、被截断、没有推理、
没有答案、500)。只用标准库;`--seq-len` 那几项另外需要 tokenizers,没有就跳过。
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
GEN = os.path.join(SRC, "gen_trace.py")
PREP = os.path.join(SRC, "prepare_corpus.py")

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


# ---------------------------------------------------------------- 假服务
RECEIVED: list[dict] = []


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # 安静
        pass

    def do_GET(self):  # noqa: N802 - 连接预检问的 /v1/models
        payload = json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode("utf-8")
        self.send_response(200 if self.path.endswith("/models") else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)
        q = body["messages"][0]["content"]
        if "CASE_500" in q:
            self.send_response(500)
            self.end_headers()
            return
        message = {"role": "assistant", "content": f"答案:{q[:12]}", "reasoning_content": f"先想一想 {q[:12]}"}
        finish = "stop"
        if "CASE_FIELD_REASONING" in q:      # 新版 vLLM 的字段名
            message = {"role": "assistant", "content": "答案B", "reasoning": "推理B"}
        elif "CASE_INLINE" in q:             # 服务端没开 reasoning parser
            message = {"role": "assistant", "content": "推理C 在 content 里</think>\n\n答案C"}
        elif "CASE_TRUNC" in q:
            finish = "length"
        elif "CASE_NOREASON" in q:
            message = {"role": "assistant", "content": "只有答案"}
        elif "CASE_NOANSWER" in q:
            message = {"role": "assistant", "content": "", "reasoning_content": "只有推理"}
        elif "CASE_LONG" in q:
            message = {"role": "assistant", "content": "答案L", "reasoning_content": "很长 " * 400}
        payload = json.dumps({
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": {"completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 20}},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run(script: str, *argv: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable] + (["-I"] if sys.flags.isolated else []) + [script, *argv]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


PAD = " —— 这是一道用来凑够最短长度的测试提问,后面这些字没有别的含义。"

# 和 Qwen3.6 的模板同一个套路的小模板:认 reasoning_content,生成提示以 <think>\n 结尾。
TEMPLATE = r"""{%- for message in messages %}
{%- if message.role == 'user' %}
{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == 'assistant' %}
{%- set rc = message.reasoning_content if message.reasoning_content is string else '' %}
{{- '<|im_start|>assistant\n<think>\n' + rc|trim + '\n</think>\n\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"""
# 不认 reasoning_content 的模板:推理会被静默丢掉
TEMPLATE_DEAF = TEMPLATE.replace(
    "{%- set rc = message.reasoning_content if message.reasoning_content is string else '' %}",
    "{%- set rc = '' %}",
)


def main() -> int:
    print("================ gen_trace 自测 ================")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

    with tempfile.TemporaryDirectory() as tmp:
        prompts = os.path.join(tmp, "prompts.jsonl")
        rows = [
            {"text": "<|User|>ASCII 标记的提问" + PAD + "<|Assistant|>参考答案,不该被用到"},
            {"text": "<｜User｜>全角标记的提问" + PAD + "<｜Assistant｜><think>旧推理</think>旧答案"},
            {"messages": [{"role": "user", "content": "messages 形状的提问" + PAD},
                          {"role": "assistant", "content": "旧答案"}]},
            {"prompt": "prompt 形状 CASE_FIELD_REASONING" + PAD},
            {"instruction": "instruction 形状 CASE_INLINE" + PAD, "input": "附加输入"},
            {"prompt": "CASE_TRUNC 会被截断" + PAD},
            {"prompt": "CASE_NOREASON 没有推理" + PAD},
            {"prompt": "CASE_NOANSWER 没有答案" + PAD},
            {"prompt": "CASE_500 服务端出错" + PAD},
            {"prompt": "CASE_LONG 推理很长" + PAD},
            {"prompt": "太短"},
            {"prompt": "prompt 形状   CASE_FIELD_REASONING" + PAD},   # 只差空白 -> 重复
            {"prompt": "这是评测集里的一道题,不能进训练语料" + PAD},
            {"nonsense": 1},
        ]
        with open(prompts, "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
            fh.write("这一行不是 JSON\n")
        evalset = os.path.join(tmp, "evalset.jsonl")
        with open(evalset, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"question": "这是评测集里的一道题,  不能进训练语料" + PAD}, ensure_ascii=False) + "\n")

        out = os.path.join(tmp, "traces.jsonl")
        common = ["--prompts", prompts, "--url", url, "--model", "fake", "--n", "0", "--par", "4",
                  "--exclude", evalset, "--out", out]
        p = run(GEN, *common)
        print("\n--- 一次完整的生成 ---")
        check("退出码 0", p.returncode == 0, p.stdout[-800:] + p.stderr[-600:])
        got = read_jsonl(out) if os.path.exists(out) else []
        by_q = {r["messages"][0]["content"]: r for r in got}
        check("写出 6 条(5 种正常情况 + 长推理)", len(got) == 6, f"{len(got)}: {sorted(by_q)}")
        check("每行是 [user, assistant],assistant 带 reasoning_content",
              all([m["role"] for m in r["messages"]] == ["user", "assistant"]
                  and r["messages"][1].get("reasoning_content") for r in got))
        check("DSv4.1 文本格式:ASCII 和全角标记都只取 User 段",
              "ASCII 标记的提问" + PAD in by_q and "全角标记的提问" + PAD in by_q
              and not any("参考答案" in q or "旧答案" in q for q in by_q))
        inst = by_q.get("instruction 形状 CASE_INLINE" + PAD + "\n\n附加输入")
        check("instruction + input 拼成一条提问", inst is not None)
        check("推理留在 content 里时按 </think> 拆开",
              inst is not None and inst["messages"][1]["reasoning_content"] == "推理C 在 content 里"
              and inst["messages"][1]["content"] == "答案C", str(inst)[:200])
        fr = by_q.get("prompt 形状 CASE_FIELD_REASONING" + PAD)
        check("认 `reasoning` 字段(新版 vLLM 的名字)",
              fr is not None and fr["messages"][1]["reasoning_content"] == "推理B")
        check("截断 / 没推理 / 没答案 / 500 的都没写出",
              not any(k in q for q in by_q for k in ("CASE_TRUNC", "CASE_NOREASON", "CASE_NOANSWER", "CASE_500")))
        check("评测集里的题被排除(只差空白也算同一道)", not any("评测集里的一道题" in q for q in by_q))
        check("没向服务请求评测题", not any("评测集里的一道题" in b["messages"][0]["content"] for b in RECEIVED))
        check("只差空白的重复提问只问一次",
              sum("CASE_FIELD_REASONING" in b["messages"][0]["content"] for b in RECEIVED) == 1)
        check("计数打出来了", all(k in p.stdout for k in ("截断 1", "没有推理 1", "没有答案 1", "请求失败 1")),
              p.stdout[-500:])
        body = RECEIVED[0]
        check("请求带上了采样参数(默认 0.6 / 0.95 / top_k 20 / max_tokens 3072)",
              (body.get("temperature"), body.get("top_p"), body.get("top_k"), body.get("max_tokens"))
              == (0.6, 0.95, 20, 3072), str({k: body.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens")}))
        check("记下了 finish_reason 和 token 数", all(r.get("finish_reason") == "stop"
                                                and r.get("reasoning_tokens") == 20 for r in got))

        print("\n--- 排除评测题:整题包含,不是全文相等,也不是只比开头 ---")
        stem = "Which of the following statements about quantization is incorrect, given the setup below?"
        ev2 = os.path.join(tmp, "evalset2.json")
        with open(ev2, "w", encoding="utf-8") as fh:
            json.dump([
                {"problem": stem + " The model has 256 experts and a hidden size of 2048."},   # MATH-500 的字段名
                {"question_content": "给定一个整数数组,返回其中两个数之和等于目标值的下标,要求时间复杂度为 O(n),并说明理由。"},  # LCB 的字段名
                {"question": "短题"},
            ], fh, ensure_ascii=False)
        pool2 = os.path.join(tmp, "pool2.jsonl")
        pool_rows = [
            # 评测题 + 选项:全文不相等,但整题在里面 -> 应剔除
            {"prompt": stem + " The model has 256 experts and a hidden size of 2048.\nA. x\nB. y\nAnswer with the letter."},
            # 只有题干开头相同(通用题干),后面是另一道题 -> 不该剔除
            {"prompt": stem + " The codebook has 16 entries and the group size is 128."},
            {"prompt": "请解答:给定一个整数数组,返回其中两个数之和等于目标值的下标,要求时间复杂度为  O(n),并说明理由。用 Python。"},
            {"prompt": "短题"},                       # 不足 40 字符的评测题:全文相等才剔
            {"prompt": "这道题里提到了短题两个字,但不是那道评测题,所以应当保留下来才对。" + PAD},
            {"prompt": "一道完全无关的提问,用来确认正常的提问不会被误伤。" + PAD},
        ]
        with open(pool2, "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in pool_rows) + "\n")
        dumped = os.path.join(tmp, "clean_prompts.jsonl")
        n_req = len(RECEIVED)
        p = run(GEN, "--prompts", pool2, "--exclude", ev2, "--n", "0", "--min-chars", "0", "--dump-prompts", dumped)
        kept_q = [r["prompt"] for r in read_jsonl(dumped)] if os.path.exists(dumped) else []
        check("--dump-prompts:退出码 0,不需要 --out / --model,也不请求服务",
              p.returncode == 0 and len(RECEIVED) == n_req, p.stdout[-500:] + p.stderr[-400:])
        check("评测题 + 选项的提问被剔除(整题包含)", not any("Answer with the letter" in q for q in kept_q), str(kept_q)[:300])
        check("只有通用题干相同的提问保留(不是只比开头)", any("codebook has 16 entries" in q for q in kept_q))
        check("中文评测题只差空白也剔除;字段名 problem / question_content 都认",
              not any("两个数之和" in q for q in kept_q))
        check("不足 40 字符的评测题只在全文相等时剔除", "短题" not in kept_q and any("提到了短题两个字" in q for q in kept_q))
        check("共留下 3 条,并报告了剔掉几条", len(kept_q) == 3 and "剔掉 3 条提问" in p.stdout, p.stdout[-400:])
        empty_ev = os.path.join(tmp, "empty_eval.jsonl")
        with open(empty_ev, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"nonsense": 1}) + "\n")
        p = run(GEN, "--prompts", pool2, "--exclude", empty_ev, "--dump-prompts", os.path.join(tmp, "x2.jsonl"))
        check("评测题文件读不出题目时停下(不在没排除的情况下继续)", p.returncode != 0, p.stdout[-300:] + p.stderr[-300:])
        p = run(GEN, "--prompts", pool2)
        check("既没给 --dump-prompts 又没给 --out / --model 时报用法错误", p.returncode != 0)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            print("  [skip] 没有 pyarrow,parquet 评测集那一项不跑")
        else:
            evdir = os.path.join(tmp, "evaldir")
            os.makedirs(evdir)
            pq.write_table(pa.table({"question": [stem + " The model has 256 experts and a hidden size of 2048."],
                                     "answer": ["A"]}), os.path.join(evdir, "test-00000-of-00001.parquet"))
            d2 = os.path.join(tmp, "clean2.jsonl")
            p = run(GEN, "--prompts", pool2, "--exclude", evdir, "--n", "0", "--min-chars", "0", "--dump-prompts", d2)
            got2 = [r["prompt"] for r in read_jsonl(d2)] if os.path.exists(d2) else []
            check("--exclude 直接给评测集目录(里面是 parquet)",
                  p.returncode == 0 and len(got2) == 5 and not any("Answer with the letter" in q for q in got2),
                  p.stdout[-400:] + p.stderr[-300:])

        print("\n--- 连接预检:连不上 / 模型名不对,第一时间说清楚 ---")
        # 2026-11-17 第一次对着真环境跑:localhost 解析成 ::1,50 条全报 OSError(99),
        # 脚本发完 50 条才说"一条都没写出来",还留下一个空文件。
        import socket
        import urllib.error

        sys.path.insert(0, SRC)
        import gen_trace as G  # noqa: E402

        lines = " ".join(G._explain(urllib.error.URLError(OSError(99, "Cannot assign requested address")),
                                     "http://localhost:8901/v1/chat/completions"))
        check("errno 99 + localhost:说明是 IPv6 回环的问题,并让换成 127.0.0.1",
              "::1" in lines and "127.0.0.1" in lines, lines)
        lines = " ".join(G._explain(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
                                     "http://127.0.0.1:8901/v1/chat/completions"))
        check("errno 111:说明是没有服务在听", "没有服务在听" in lines, lines)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
        sock.close()                               # 这个端口现在没人听
        dead_out = os.path.join(tmp, "dead.jsonl")
        n_req = len(RECEIVED)
        p = run(GEN, "--prompts", prompts, "--model", "fake", "--n", "0", "--out", dead_out,
                "--url", f"http://127.0.0.1:{dead_port}/v1/chat/completions")
        check("服务没起来:直接停,退出码非 0,说明原因",
              p.returncode != 0 and "连不上服务" in (p.stdout + p.stderr), p.stdout[-300:] + p.stderr[-400:])
        check("服务没起来:一条提问都没发,也没留下输出文件",
              len(RECEIVED) == n_req and not os.path.exists(dead_out))
        p = run(GEN, "--prompts", prompts, "--url", url, "--model", "qwen36", "--n", "0", "--out", dead_out)
        check("模型名不对:直接停,并列出服务里有的模型名",
              p.returncode != 0 and "fake" in (p.stdout + p.stderr) and not os.path.exists(dead_out),
              p.stdout[-300:] + p.stderr[-400:])

        only500 = os.path.join(tmp, "only500.jsonl")
        with open(only500, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"prompt": "CASE_500 服务端出错" + PAD}, ensure_ascii=False) + "\n")
        out500 = os.path.join(tmp, "out500.jsonl")
        p = run(GEN, "--prompts", only500, "--url", url, "--model", "fake", "--n", "0", "--out", out500)
        check("请求全部失败:退出码非 0,说明是服务的问题,不留空文件,也不打印「下一步」",
              p.returncode != 0 and "所有请求都失败" in (p.stdout + p.stderr)
              and not os.path.exists(out500) and "下一步" not in p.stdout, p.stdout[-400:] + p.stderr[-300:])

        empty_out = os.path.join(tmp, "empty_out.jsonl")
        open(empty_out, "w").close()
        p = run(GEN, "--prompts", prompts, "--url", url, "--model", "fake", "--n", "0", "--exclude", evalset,
                "--out", empty_out)
        check("--out 是个空文件时不用 --resume 也能跑(上次失败留下的)",
              p.returncode == 0 and len(read_jsonl(empty_out)) == 6, p.stdout[-300:] + p.stderr[-300:])

        print("\n--- 断点续跑 ---")
        p = run(GEN, *common)
        check("--out 已存在且没给 --resume 时拒绝", p.returncode != 0 and "--resume" in (p.stdout + p.stderr))
        n_before = len(RECEIVED)
        p = run(GEN, *common, "--resume")
        asked = [b["messages"][0]["content"] for b in RECEIVED[n_before:]]
        check("--resume 不重问已有的 6 条", p.returncode == 0 and not any(q in by_q for q in asked),
              p.stdout[-400:] + p.stderr[-300:])
        check("--resume 之后文件里还是 6 条(没写重)", len(read_jsonl(out)) == 6)

        print("\n--- --keep-truncated ---")
        out2 = os.path.join(tmp, "traces_trunc.jsonl")
        p = run(GEN, "--prompts", prompts, "--url", url, "--model", "fake", "--n", "0", "--out", out2,
                "--keep-truncated")
        kept = read_jsonl(out2) if os.path.exists(out2) else []
        check("保留截断的轨迹,finish_reason 记为 length",
              p.returncode == 0 and any(r["finish_reason"] == "length" for r in kept), p.stdout[-300:])
        check("没给 --exclude 时明说没查重叠", "没给 --exclude" in p.stdout)

        print("\n--- 产物过 prepare_corpus ---")
        tokdir = os.path.join(tmp, "tok")
        os.makedirs(tokdir)
        with open(os.path.join(tokdir, "chat_template.jinja"), "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE)
        corpus = os.path.join(tmp, "corpus.jsonl")
        p = run(PREP, "--tokenizer", tokdir, "--src", out, "--out", corpus)
        check("prepare_corpus 退出码 0", p.returncode == 0, p.stdout[-800:] + p.stderr[-500:])
        rows_c = read_jsonl(corpus) if os.path.exists(corpus) else []
        check("6 条全部保留,推理没丢", len(rows_c) == 6
              and all(r["messages"][1].get("reasoning_content") for r in rows_c), p.stdout[-300:])
        check("输出只有 messages(生成时的统计字段不带进训练语料)", all(list(r) == ["messages"] for r in rows_c))
        check("报告了带推理的条数", "带推理:      6" in p.stdout, p.stdout[-400:])

        mixed = os.path.join(tmp, "mixed.jsonl")
        with open(mixed, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"prompt": "没有推理的一条", "response": "答"}, ensure_ascii=False) + "\n")
        corpus_m = os.path.join(tmp, "corpus_mixed.jsonl")
        p = run(PREP, "--tokenizer", tokdir, "--src", mixed, "--src", out, "--out", corpus_m)
        rows_m = read_jsonl(corpus_m) if os.path.exists(corpus_m) else []
        check("带推理和不带推理的混在一起时,每条 assistant 都有 reasoning_content 键(不带的是空串)",
              p.returncode == 0 and len(rows_m) == 7
              and all("reasoning_content" in r["messages"][1] for r in rows_m)
              and rows_m[0]["messages"][1]["reasoning_content"] == "", p.stdout[-300:] + p.stderr[-300:])
        plain = os.path.join(tmp, "corpus_plain.jsonl")
        p = run(PREP, "--tokenizer", tokdir, "--src", mixed, "--out", plain)
        check("全都不带推理时不写 reasoning_content(老格式不变)",
              p.returncode == 0 and all("reasoning_content" not in m for r in read_jsonl(plain) for m in r["messages"]))

        deaf = os.path.join(tmp, "tok_deaf")
        os.makedirs(deaf)
        with open(os.path.join(deaf, "chat_template.jinja"), "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE_DEAF)
        p = run(PREP, "--tokenizer", deaf, "--src", out, "--out", os.path.join(tmp, "x.jsonl"))
        check("模板不认 reasoning_content 时拒绝(否则推理被静默丢掉)",
              p.returncode != 0 and "reasoning_content" in (p.stdout + p.stderr), p.stdout[-300:] + p.stderr[-300:])

        print("\n--- --seq-len:丢掉训练装不下的 ---")
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError:
            print("  [skip] 没有 tokenizers 包,这一组不跑")
        else:
            tk = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
            tk.pre_tokenizer = pre_tokenizers.Whitespace()
            tk.save(os.path.join(tokdir, "tokenizer.json"))
            corpus_s = os.path.join(tmp, "corpus_seq.jsonl")
            p = run(PREP, "--tokenizer", tokdir, "--src", out, "--seq-len", "200", "--out", corpus_s)
            rows_s = read_jsonl(corpus_s) if os.path.exists(corpus_s) else []
            check("超过 seq_len 的那条(长推理)被丢掉,其余 5 条保留",
                  p.returncode == 0 and len(rows_s) == 5
                  and not any("CASE_LONG" in r["messages"][0]["content"] for r in rows_s),
                  p.stdout[-500:] + p.stderr[-300:])
            check("报告了丢了几条和 token 合计", "超过 seq_len: 1" in p.stdout and "token 合计" in p.stdout,
                  p.stdout[-500:])

    server.shutdown()
    print(f"\n================ {_ok} 项通过, {_fail} 项失败 ================")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
