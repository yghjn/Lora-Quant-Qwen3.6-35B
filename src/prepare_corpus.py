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


def render_check(template: str, messages: list[dict]) -> str:
    """用 jinja2 渲一遍。上游 set_chat_template 也是 ``env.from_string(template)``。"""
    try:
        from jinja2 import Environment, StrictUndefined
    except ImportError:
        _die("缺 jinja2。", "它是 torch 的依赖,两个 env 里都该有:pip install jinja2")
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    return env.from_string(template).render(messages=messages)


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
            out.append({"role": role, "content": content})
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

    if not kept:
        _die("一条样本都没有。", "给 --src,或者用 --synthetic N 生成内置样本跑冒烟。")

    # 每条都真渲一遍 —— 工单 §6.6 的验收是"chat template 成功必须 100%,回退 0"。
    failed = 0
    for msgs in kept:
        try:
            text = render_check(template, msgs)
        except Exception:  # noqa: BLE001 - 渲染失败的原因不重要,数量重要
            failed += 1
            continue
        if not text.strip():
            failed += 1

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
    print(f"超长丢弃:    {too_long}")
    print(f"内置补充:    {args.synthetic}")
    print(f"写出:        {len(kept)}  -> {args.out}")
    print(f"角色分布:    {roles}")
    print(f"总字符:      {chars}  (平均 {chars / len(kept):.0f}/条)")
    print(f"chat template 成功: {len(kept) - failed}/{len(kept)}   回退/失败: {failed}")

    if failed:
        _die(f"{failed} 条渲染失败 —— 工单 §6.6 的验收要求是 0。")

    print("\n下一步:")
    print(f"    export DATA_FILES={os.path.abspath(args.out)}")
    print("    MODULE=qwen36_lora_recipe CONFIG=qwen36_35b_a3b_lora_smoke ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
