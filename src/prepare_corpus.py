#!/usr/bin/env python
"""把指令数据整成 ChatDataLoader 吃的 jsonl,并**先把 chat template 验死**(工单 §6.6)。

    python -I src/prepare_corpus.py --tokenizer <tokenizer 目录> --synthetic 64 \
        --out corpus/smoke.jsonl
    python -I src/prepare_corpus.py --tokenizer <tokenizer 目录> \
        --src data/alpaca.json --src data/more.jsonl --out corpus/sft.jsonl

产物每行一个 ``{"messages": [{"role": ..., "content": ...}, ...]}``,
正好是 ``torchtitan_npu.models.qwen3_5.config_registry._messages`` 要的形状。

----

**这个脚本存在的唯一理由是 DSv4.1 那个有实测代价的坑。**
他们两个模型目录的 ``tokenizer.chat_template`` 根本没设,自写的 ``gen_trace.py``
一路静默回退成 5 个 ASCII 字面 token,修好后 LoRA 造成的退化从 +13.1 点压到 +3.5 点。

**好消息:训练这条路本身是会响的。** 上游
``torchtitan/components/tokenizer.py:89-90`` 的 ``apply_chat_template`` 在没有
模板时直接 ``raise ValueError("No chat template set.")``,而
``hf_datasets/text_datasets.py:489`` 就是调它。所以静默回退只会发生在**自己写的
数据生成脚本**里,不会发生在 ChatDataLoader 里。

那这里还验什么:验的是**第一步就炸,而不是第 1 个 batch 才炸**。
模板缺失在 dataloader 里抛出来时,8 张卡的模型已经加载完了 —— 几分钟白等,
而且异常栈在 worker 里不一定好读。这个脚本纯 CPU、一秒钟,放在上卡之前。

它同时报告**模板是从哪个文件读到的**。上游 tokenizer 的查找顺序是
``chat_template.jinja`` 优先,其次 ``tokenizer_config.json`` 里的
``chat_template`` 键(tokenizer.py:158-160)。两个都有且内容不同时,
训练用的是前者 —— 这种目录见过,所以这里显式打出来。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 上游的查找顺序,见 components/tokenizer.py:111 和 158-160。
TEMPLATE_FILE = "chat_template.jinja"
TEMPLATE_CONFIG_FILES = ("tokenizer_config.json", "config.json")

_SYNTHETIC = [
    ("把下面这句话改写得更简洁:模型的权重在量化之后数值精度会有所下降。", "量化会降低权重精度。"),
    ("列出三种常见的权重量化方法。", "逐张量对称量化、逐通道量化、分组量化。"),
    ("What does LDLQ stand for in weight quantization?", "It is LDL-decomposition-guided quantization."),
    ("用一句话说明 LoRA 为什么能补偿量化误差。", "低秩增量可以拟合量化带来的结构化残差。"),
]


def _die(msg: str, *lines: str) -> None:
    print(f"\n!! {msg}", file=sys.stderr)
    for line in lines:
        print(f"   {line}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- chat template
def find_chat_template(tokenizer_dir: str) -> tuple[str, str]:
    """返回 (模板源文件路径, 模板字符串)。找不到就退出。"""
    path = os.path.join(tokenizer_dir, TEMPLATE_FILE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return path, fh.read()

    for name in TEMPLATE_CONFIG_FILES:
        cfg_path = os.path.join(tokenizer_dir, name)
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        tmpl = cfg.get("chat_template") if isinstance(cfg, dict) else None
        if isinstance(tmpl, str) and tmpl.strip():
            return cfg_path, tmpl

    _die(
        f"{tokenizer_dir} 里没有 chat template —— 这正是 DSv4.1 踩过的那个坑。",
        f"找过 {TEMPLATE_FILE},以及 {', '.join(TEMPLATE_CONFIG_FILES)} 里的 chat_template 键。",
        "不要自己拼模板:从官方发布的 tokenizer 目录取 chat_template.jinja 放进去,",
        "否则训练时 tokenizer.py:90 会抛 'No chat template set.',而那时 8 卡已经加载完了。",
    )
    raise AssertionError("unreachable")


def _config_has_template(tokenizer_dir: str) -> bool:
    """tokenizer_config.json / config.json 里是否也有 chat_template 键。

    只用来提示"两处都有、训练用的是哪一个",所以读不动就当没有,不报错。
    """
    for name in TEMPLATE_CONFIG_FILES:
        path = os.path.join(tokenizer_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("chat_template"), str):
            return True
    return False


_COMPILED: dict[str, object] = {}


def _compile(template: str):
    """按上游 ``BaseTokenizer.set_chat_template`` 的方式编译模板 —— **逐项照抄**。

    出处:``torchtitan/components/tokenizer.py:43-78``(upstream 086bf6c16)。
    这里验的是"训练时会不会炸",所以环境必须和训练时一样,不能更严也不能更松:

    * ``ImmutableSandboxedEnvironment``,``trim_blocks`` / ``lstrip_blocks`` 都开,
      扩展只有 ``loopcontrols``;
    * **默认的 Undefined(宽松)**。2026-11-16 以前这里用的是 ``StrictUndefined``,
      于是 Qwen3.6 模板里的 ``{%- if tools ... %}`` 在没传 ``tools`` 时直接
      ``UndefinedError`` —— 训练根本不会这样,是这个检查自己比训练严;
    * 全局函数 ``raise_exception`` / ``strftime_now``,过滤器 ``tojson``。

    模板编译不过就退出。最常见的原因见下面的报错说明。
    """
    if template in _COMPILED:
        return _COMPILED[template]
    try:
        import jinja2
        import jinja2.ext
        import jinja2.sandbox
    except ImportError:
        _die("缺 jinja2。", "它是 torch 的依赖,两个 env 里都该有:pip install jinja2")

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )

    def strftime_now(fmt):
        from datetime import datetime

        return datetime.now().strftime(fmt)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = tojson
    try:
        compiled = env.from_string(template)
    except jinja2.exceptions.TemplateSyntaxError as exc:
        lines = [
            f"第 {exc.lineno} 行: {exc.message}",
            "训练时上游建 tokenizer 用的是同一套 jinja 环境,会在同一处报同样的错。",
        ]
        if "generation" in str(exc.message):
            lines += [
                "模板里有 `{%- generation %}` / `{%- endgeneration %}`:这是 HF transformers 专用的",
                "标签(只用来标 assistant 段,不产生任何文本),上游 torchtitan 的 jinja 环境不认。",
                "Hub 上较新版本的 Qwen3.6 模板带这两行,较早的版本没有。",
                "处理:在一份**可写的** checkpoint 目录副本里把这两行删掉(权重分片可以用软链接)。",
            ]
        _die("chat template 编译不过。", *lines)
    _COMPILED[template] = compiled
    return compiled


def render_check(template: str, messages: list[dict], **kwargs) -> str:
    """渲一遍。和上游 ``apply_chat_template`` 一样:``render(messages=..., **kwargs)``。"""
    return _compile(template).render(messages=messages, **kwargs)


def single_turn_problem(messages: list[dict]) -> str | None:
    """上游 ChatDataset 只收单轮 ``[user, assistant]``,这里提前拦。

    出处:``hf_datasets/text_datasets.py`` 的 ``_validate_messages``(upstream
    086bf6c16),不满足时在**第一个 batch** 抛 ValueError —— 那时 8 卡已经加载完了。
    返回 None 表示没问题,否则返回原因。
    """
    if len(messages) != 2:
        return f"要恰好 2 条消息(一问一答),现在是 {len(messages)} 条"
    if messages[0]["role"] != "user":
        return f"第 1 条必须是 user,现在是 {messages[0]['role']!r}"
    if messages[1]["role"] != "assistant":
        return f"第 2 条必须是 assistant,现在是 {messages[1]['role']!r}"
    return None


# ---------------------------------------------------------------- 归一化
def normalize(sample) -> list[dict] | None:
    """把常见的几种指令数据形状折成 messages。认不出来就返回 None。"""
    if not isinstance(sample, dict):
        return None

    msgs = sample.get("messages") or sample.get("conversations")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                return None
            role = m.get("role") or m.get("from")
            content = m.get("content") if "content" in m else m.get("value")
            if not isinstance(role, str) or not isinstance(content, str):
                return None
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            item = {"role": role, "content": content}
            # 自生成轨迹(gen_trace.py)把模型的推理放在 reasoning_content 里。
            # Qwen 的模板认这个字段,会把推理渲进 <think> 和 </think> 之间,后面跟答案。
            reasoning = m.get("reasoning_content")
            if role == "assistant" and isinstance(reasoning, str) and reasoning.strip():
                item["reasoning_content"] = reasoning
            out.append(item)
        return out

    if isinstance(sample.get("prompt"), str) and isinstance(sample.get("response"), str):
        return [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]

    # alpaca 系:instruction / input / output
    if isinstance(sample.get("instruction"), str) and isinstance(sample.get("output"), str):
        user = sample["instruction"]
        extra = sample.get("input")
        if isinstance(extra, str) and extra.strip():
            user = f"{user}\n\n{extra}"
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": sample["output"]},
        ]

    return None


def read_source(path: str):
    """读 .jsonl(每行一个对象)或 .json(一个数组)。"""
    with open(path, encoding="utf-8") as fh:
        head = fh.read(1)
        fh.seek(0)
        if head == "[":
            data = json.load(fh)
            if not isinstance(data, list):
                _die(f"{path} 的顶层不是数组。")
            yield from data
        else:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    _die(f"{path}:{lineno} 不是合法 JSON。", str(exc))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", required=True, help="含 chat_template 的 tokenizer 目录")
    ap.add_argument("--src", action="append", default=[], help="输入数据,可重复给")
    ap.add_argument("--synthetic", type=int, default=0, help="再补 N 条内置样本(给冒烟用)")
    ap.add_argument("--out", help="输出 jsonl;不给就只做模板检查")
    ap.add_argument("--max-chars", type=int, default=0, help=">0 时丢掉超长样本")
    ap.add_argument(
        "--seq-len", type=int, default=0,
        help=">0 时按 token 数丢掉训练装不下的样本(上游 ChatDataset 对超过 seq_len 的样本是"
             "直接丢弃,不截断)。填训练配置的 seq_len,现在是 4096。需要 tokenizers 包",
    )
    ap.add_argument("--skip-bad", action="store_true", help="认不出来的样本跳过而不是报错退出")
    args = ap.parse_args()

    print("=== 1. chat template ===")
    src_path, template = find_chat_template(args.tokenizer)
    print(f"模板来源: {src_path}")
    print(f"模板长度: {len(template)} 字符")

    if os.path.isfile(os.path.join(args.tokenizer, TEMPLATE_FILE)) and _config_has_template(
        args.tokenizer
    ):
        print(
            f"注意: {TEMPLATE_FILE} 和 tokenizer_config.json 里都有模板。"
            f"上游优先用前者(tokenizer.py:158),本次检查也用前者。"
        )

    probe = [
        {"role": "user", "content": "一加一等于几?"},
        {"role": "assistant", "content": "二。"},
    ]
    rendered = render_check(template, probe)
    print("渲染样例(前 400 字符):")
    print("-" * 60)
    print(rendered[:400])
    print("-" * 60)

    # 上游用"只渲 user 那一条 + add_generation_prompt=True"的 token 数来定 loss 从哪
    # 开始算(text_datasets.py 的 _tokenize_sample)。它应当是整段的前缀;不是的话
    # 边界会偏几个 token。只提示,不拦 —— 上游自己也只按 token 数切。
    prompt = render_check(template, probe[:1], add_generation_prompt=True)
    tail = rendered[len(prompt):] if rendered.startswith(prompt) else None
    if tail is None:
        print("注意: prompt 段不是整段的前缀 —— loss 的起点会偏。把上面的样例和下面这段对一下:")
        print(repr(prompt[-200:]))
    else:
        print(f"prompt 段到这里为止(之前的不算 loss): ...{prompt[-40:]!r}")
        print(f"算 loss 的部分(模型要学着生成的): {tail.rstrip(chr(10))!r}")
    if "一加一等于几?" not in rendered or "二。" not in rendered:
        _die(
            "渲染结果里找不到输入内容 —— 模板没把 messages 放进去。",
            "这正是 DSv4.1 静默回退的症状。先确认模板文件是对的那份。",
        )
    print("模板检查: 通过")

    if not args.out:
        print("\n没给 --out,只做模板检查,不写文件。")
        return 0

    print("\n=== 2. 归一化 ===")
    kept: list[list[dict]] = []
    bad = 0
    too_long = 0
    not_single = 0
    seen = 0
    for path in args.src:
        if not os.path.isfile(path):
            _die(f"找不到输入文件 {path}")
        for sample in read_source(path):
            seen += 1
            msgs = normalize(sample)
            if msgs is None:
                bad += 1
                continue
            why = single_turn_problem(msgs)
            if why is not None:
                if not_single == 0:
                    print(f"第一条不是单轮问答的样本(第 {seen} 条): {why}")
                not_single += 1
                continue
            if args.max_chars and sum(len(m["content"]) for m in msgs) > args.max_chars:
                too_long += 1
                continue
            kept.append(msgs)

    for i in range(args.synthetic):
        user, assistant = _SYNTHETIC[i % len(_SYNTHETIC)]
        kept.append(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        )

    if bad and not args.skip_bad:
        _die(
            f"{bad}/{seen} 条样本认不出形状。",
            "支持的形状:messages / conversations / prompt+response / instruction+input+output。",
            "确认要丢掉它们就加 --skip-bad。",
        )

    if not_single and not args.skip_bad:
        _die(
            f"{not_single}/{seen} 条样本不是单轮 [user, assistant]。",
            "上游 ChatDataset 只收这一种形状(text_datasets.py 的 _validate_messages),",
            "多轮对话、带 system 的样本会在第一个 batch 抛 ValueError。",
            "确认要丢掉它们就加 --skip-bad。",
        )

    if not kept:
        _die("一条样本都没有。", "给 --src,或者用 --synthetic N 生成内置样本跑冒烟。")

    # 数 token 用的 tokenizer。数的是"这条样本在训练的一行里占几个位置",
    # 算法照上游 text_datasets.py 的 _tokenize_sample,见下面循环里的注释。
    tok = None
    eos_id = None
    if args.seq_len > 0:
        tok_path = os.path.join(args.tokenizer, "tokenizer.json")
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(tok_path)
            eos_id = tok.token_to_id("<|im_end|>")
        except ImportError:
            _die("--seq-len 需要 tokenizers 包。", "训练环境里有;或者去掉 --seq-len(那样就不数 token 了)。")
        except Exception as exc:  # noqa: BLE001
            _die(f"读不了 {tok_path}", str(exc))

    # 每条都真渲一遍 —— 工单 §6.6 的验收是"chat template 成功必须 100%,回退 0"。
    failed = 0
    rendered_total = len(kept)
    with_reasoning = 0
    reasoning_lost = 0
    over_len = 0
    ntoks: list[int] = []
    fitted: list[list[dict]] = []
    for msgs in kept:
        try:
            text = render_check(template, msgs)
        except Exception:  # noqa: BLE001 - 渲染失败的原因不重要,数量重要
            failed += 1
            continue
        if not text.strip():
            failed += 1
            continue
        reasoning = msgs[-1].get("reasoning_content")
        if reasoning:
            with_reasoning += 1
            # 模板不认 reasoning_content 的话,推理会被静默丢掉,只剩一个空的思考块
            if reasoning.strip()[:200] not in text:
                reasoning_lost += 1
        if tok is not None:
            # 上游(_tokenize_sample):整段去掉结尾换行后编码;结尾不是 EOS 就补一个;
            # 输入是去掉最后一个 token 的那一段,它的长度超过 seq_len 就丢。
            # Qwen 的 EOS 是 <|im_end|>,模板渲出来本来就以它结尾,所以输入长度 = 编码长度 - 1。
            # 2026-11-19 用上游真的 ChatDataset 逐条对过(verify/probe_steps_upstream_cpu.py);
            # 之前这里按"编码长度"算,每条多数了 1。
            ids = tok.encode(text.rstrip(chr(10))).ids
            n = len(ids) - 1 if (eos_id is not None and ids and ids[-1] == eos_id) else len(ids)
            if n > args.seq_len:
                over_len += 1
                continue
            ntoks.append(n)
        fitted.append(msgs)
    kept = fitted

    if reasoning_lost:
        _die(
            f"{reasoning_lost}/{with_reasoning} 条样本的 reasoning_content 没出现在渲染结果里。",
            "这份 chat template 不认 reasoning_content —— 推理会被静默丢掉,训练时只剩空的思考块。",
            "确认用的是 Qwen3.6 自己的模板。",
        )
    if not kept:
        _die("一条样本都没剩下。", f"渲染失败 {failed} 条,超过 --seq-len 的 {over_len} 条。")

    # 有任何一条带推理时,给每条 assistant 消息都写上 reasoning_content(没有的写空串)。
    # datasets 读 json 是按块推断结构的,同一个字段有的行有、有的行没有,出现在后面的块里
    # 会报结构不一致。空串渲出来和不带这个字段完全一样(都是空的思考块)。
    if with_reasoning:
        for msgs in kept:
            msgs[-1].setdefault("reasoning_content", "")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for msgs in kept:
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    roles: dict[str, int] = {}
    chars = 0
    for msgs in kept:
        for m in msgs:
            roles[m["role"]] = roles.get(m["role"], 0) + 1
            chars += len(m["content"])

    print(f"读入:        {seen}")
    print(f"认不出:      {bad}" + ("  (已跳过)" if args.skip_bad else ""))
    print(f"不是单轮:    {not_single}" + ("  (已跳过)" if args.skip_bad and not_single else ""))
    print(f"超长丢弃:    {too_long}")
    if args.seq_len > 0:
        print(f"超过 seq_len: {over_len}  (token 数 > {args.seq_len},训练时会被上游直接丢弃,这里先丢掉)")
    print(f"带推理:      {with_reasoning}" + ("" if with_reasoning else "  (都是「空思考块 + 答案」,即关思考的格式)"))
    print(f"内置补充:    {args.synthetic}")
    print(f"写出:        {len(kept)}  -> {args.out}")
    print(f"角色分布:    {roles}")
    print(f"总字符:      {chars}  (平均 {chars / len(kept):.0f}/条)")
    print(f"chat template 成功: {rendered_total - failed}/{rendered_total}   回退/失败: {failed}")

    if ntoks:
        v = sorted(ntoks)
        n = len(v)
        print(f"token 数/条: min {v[0]} p25 {v[n // 4]} 中位 {v[n // 2]} p75 {v[3 * n // 4]} max {v[-1]}")
        print(f"token 合计:  {sum(v) / 1e6:.2f} M")

    if failed:
        _die(f"{failed} 条渲染失败 —— 工单 §6.6 的验收要求是 0。")

    # 正式训练的步数是按语料量算的(qwen36_steps.py):把每条的 token 数记在语料旁边,
    # 训练配置读它。没数 token(没给 --seq-len)就没有这份记录。
    print("\n=== 3. 正式训练的步数 ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qwen36_steps as S

    if ntoks:
        meta = S.write_meta(args.out, ntoks, args.seq_len)
        print(f"每条的 token 数已记到: {meta}")
        try:
            epochs = S.epochs_from_env()
            p = S.plan_from_ntoks(ntoks, args.seq_len, S.DEFAULT_WORLD, epochs)
            print(S.describe(p))
            print(f"(训练时一行不是一条样本:上游把样本拼满 {args.seq_len} token 再训。"
                  f"遍数用环境变量 {S.EPOCHS_ENV} 改,默认 {S.DEFAULT_EPOCHS:g};"
                  f"或者命令末尾加 --training.steps N 直接定步数)")
            if p["tokens"] < 2_000_000:
                print(f"注意: token 合计 {p['tokens'] / 1e6:.2f} M,低于建议的 2 M。"
                      "跑冒烟够用;正式训练会把这点语料反复过,容易学成背答案。")
        except S.StepsError as exc:
            print(f"注意: 步数算不出来 —— {exc}")
    else:
        stale = S.meta_path(args.out)
        if os.path.isfile(stale):
            os.remove(stale)
        print("没给 --seq-len,没数 token:冒烟配置不受影响(它是固定步数);")
        print("正式配置(lora_r16 / lora_r4)会拒绝启动,除非命令末尾加 --training.steps N。")

    print("\n下一步:")
    print(f"    export DATA_FILES={os.path.abspath(args.out)}")
    print("    MODULE=qwen36_lora_recipe CONFIG=qwen36_35b_a3b_lora_smoke ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
