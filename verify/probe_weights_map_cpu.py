#!/usr/bin/env python3
"""加载 70 GB 之前,先核对 checkpoint 里的权重和模型参数**一一对得上**。

    python3 verify/probe_weights_map_cpu.py --hf-assets <checkpoint 目录>

纯 CPU、单进程、几十秒。只读 `model.safetensors.index.json` 和各分片的**文件头**
(每个几百 KB),不读权重数据。模型建在 meta device 上,不占内存。

为什么要它
----------
训练时权重是这样进模型的(upstream `components/checkpointer/dcp.py` 的 `dcp_load`)::

    hf_state_dict = sd_adapter.to_hf(model.state_dict())     # 模型要哪些 HF 键
    dcp.load(hf_state_dict, HuggingFaceStorageReader(dir))   # 从分片里读
    model.load_state_dict(sd_adapter.from_hf(hf_state_dict), strict=False)

`Qwen35StateDictAdapter` 遇到映射表里没有的键是**直接跳过**的,最后一步又是
`strict=False`。所以如果有哪个模型参数没被映射到,它会保持随机初始化,
**不报错、不告警、日志里也看不出来** —— 训练照常跑,只是 loss 偏高。
这个脚本把那三步里"键和形状"的部分原样走一遍,把跳过的东西摆出来。

判据
----
0. 专家权重的排布语义和 vLLM 的读法一致(小张量对拍,不依赖 checkpoint);
1. 模型要的每个 HF 键,checkpoint 里都有(否则 `dcp.load` 会报缺键);
2. 每个**非 adapter** 的模型参数都会被加载(没被加载的逐个列出);
3. 形状一致(读各分片文件头;目录里没有分片文件时跳过这一条并说明);
4. checkpoint 里没被用到的键只属于已知可忽略的几类(`mtp.*` 等),其余逐个列出。

adapter 参数(名字含 `lora_`)不在 checkpoint 里是**预期的**:它们从零初始化开始训。

退出码:0 通过;1 判据不成立;2 环境 / 参数问题。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# checkpoint 里有、但模型本来就不用的键。mtp.* 是多 token 预测头,torchtitan 的
# Qwen3.5 模型没有这个分支;训练和 vLLM 推理都不走它。
IGNORABLE_CKPT_PREFIXES = ("mtp.",)


def _pattern(key: str) -> str:
    return re.sub(r"\d+", "{}", key)


def _summarize(keys, limit: int = 12) -> list[str]:
    counts = collections.Counter(_pattern(k) for k in keys)
    lines = [f"{n:>5} × {pat}" for pat, n in sorted(counts.items())]
    return lines[:limit] + ([f"      ... 还有 {len(lines) - limit} 类"] if len(lines) > limit else [])


def read_safetensors_header(path: str) -> dict:
    """safetensors 文件头:8 字节小端长度 + JSON。只读头,不读数据。"""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n <= 0 or n > 200 * 1024 * 1024:
            raise ValueError(f"{path}: 文件头长度不合理({n})")
        return json.loads(fh.read(n).decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hf-assets", required=True, help="HF 格式的 checkpoint 目录(就是 HF_ASSETS_PATH)")
    ap.add_argument("--config", default="qwen36_35b_a3b_lora_smoke")
    args = ap.parse_args()

    assets = os.path.abspath(args.hf_assets)
    index_path = os.path.join(assets, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print(f"!! 找不到 {index_path}")
        return 2
    with open(index_path, encoding="utf-8") as fh:
        weight_map: dict[str, str] = json.load(fh)["weight_map"]
    print(f"checkpoint: {assets}")
    print(f"索引: {len(weight_map)} 个键,{len(set(weight_map.values()))} 个分片")

    os.environ["HF_ASSETS_PATH"] = assets
    os.environ.setdefault("DATA_FILES", "/nonexistent-corpus.jsonl")
    os.environ.pop("INITIAL_LOAD_PATH", None)

    try:
        import torch
        from torchtitan.config import TORCH_DTYPE_MAP, ConfigManager
        from torchtitan.tools import utils
    except ModuleNotFoundError as exc:
        traceback.print_exc()
        print(f"!! 缺模块 {exc.name} —— 先让 env/2_pip_train.sh 的闸门过了。")
        return 2

    config = ConfigManager().parse_args(["--module", "qwen36_lora_recipe", "--config", args.config])
    spec = config.model_spec
    model_config = spec.model
    model_config.update_from_config(config=config)
    with torch.device("meta"), utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
        model = model_config.build()
    adapter = spec.state_dict_adapter(model_config, assets)

    state = dict(model.state_dict())
    adapters = [k for k in state if "lora_" in k]
    base = {k: v for k, v in state.items() if "lora_" not in k}
    print(f"模型: {len(state)} 个 state 键,其中 adapter {len(adapters)} 个(不从 checkpoint 加载,预期)")

    hf_wanted = adapter.to_hf(dict(state))
    back = adapter.from_hf(dict(hf_wanted))

    ok = True

    def check(name: str, cond: bool, detail: str = "", lines: list[str] | None = None) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'OK ' if cond else 'X  '} {name}" + (f": {detail}" if detail else ""))
        for line in lines or []:
            print("        " + line)

    print("\n================ 判据 ================")
    # 0. 排布语义:用几个数就能看出来的小张量走一遍适配器,和 vLLM 的读法对拍
    #    (gate_up_proj 沿 dim -2 对半切,前一半 gate=w1,后一半 up=w3;down_proj 不转置)。
    e_n, f_n, d_n = 2, 3, 4
    pre = "layers.0.moe.routed_experts.inner_experts."
    w1 = torch.arange(e_n * f_n * d_n, dtype=torch.float32).reshape(e_n, f_n, d_n)
    w3 = w1 + 1000
    w2 = torch.arange(e_n * d_n * f_n, dtype=torch.float32).reshape(e_n, d_n, f_n) + 5000
    tiny = adapter.to_hf({pre + "w1_EFD": w1, pre + "w3_EFD": w3, pre + "w2_EDF": w2})
    gu = tiny.get("model.language_model.layers.0.mlp.experts.gate_up_proj")
    dn = tiny.get("model.language_model.layers.0.mlp.experts.down_proj")
    layout_ok = (
        gu is not None and dn is not None
        and tuple(gu.shape) == (e_n, 2 * f_n, d_n) and tuple(dn.shape) == (e_n, d_n, f_n)
        and torch.equal(gu[:, :f_n, :], w1) and torch.equal(gu[:, f_n:, :], w3) and torch.equal(dn, w2)
    )
    # upstream 的 from_hf 要求 lm_head / embed 至少有一个,给个占位的
    rt = adapter.from_hf({**tiny, "lm_head.weight": torch.zeros(1)})
    rt_ok = all(k in rt and torch.equal(rt[k], v) for k, v in
                ((pre + "w1_EFD", w1), (pre + "w3_EFD", w3), (pre + "w2_EDF", w2)))
    check("0. 专家权重排布:gate_up_proj=[E,2F,D](前半 gate、后半 up),down_proj=[E,D,F],往返还原",
          layout_ok and rt_ok,
          f"gate_up {tuple(gu.shape) if gu is not None else None} down {tuple(dn.shape) if dn is not None else None}"
          f" 适配器={type(adapter).__name__}")

    missing = [k for k in hf_wanted if k not in weight_map]
    check("1. 模型要的每个 HF 键,checkpoint 里都有", not missing,
          f"要 {len(hf_wanted)} 个,缺 {len(missing)} 个", _summarize(missing))

    not_loaded = [k for k in base if k not in back]
    check("2. 每个非 adapter 的模型参数都会被加载", not not_loaded,
          f"{len(base) - len(not_loaded)} / {len(base)}", _summarize(not_loaded))

    leaked = [k for k in adapters if k in back]
    check("   adapter 参数不会被 checkpoint 覆盖", not leaked, f"{len(adapters)} 个 adapter 键", _summarize(leaked))

    shape_back = [k for k in base if k in back and tuple(back[k].shape) != tuple(base[k].shape)]
    check("   to_hf -> from_hf 往返后形状不变", not shape_back, "", _summarize(shape_back))

    shards = sorted(set(weight_map.values()))
    have = [s for s in shards if os.path.isfile(os.path.join(assets, s))]
    if len(have) != len(shards):
        print(f"  --  3. 形状核对跳过: 目录里只有 {len(have)} / {len(shards)} 个分片文件"
              "(只有索引时属正常;在真 checkpoint 目录上跑才有这一条)")
    else:
        header_shapes: dict[str, tuple[int, ...]] = {}
        for shard in shards:
            for key, meta in read_safetensors_header(os.path.join(assets, shard)).items():
                if key != "__metadata__":
                    header_shapes[key] = tuple(meta["shape"])
        bad = []
        for key, value in hf_wanted.items():
            got = header_shapes.get(key)
            if got is not None and got != tuple(value.shape):
                bad.append(f"{key}: checkpoint {got} != 模型要的 {tuple(value.shape)}")
        absent = [k for k in hf_wanted if k in weight_map and k not in header_shapes]
        check("3. 形状一致(读各分片文件头)", not bad and not absent,
              f"核对 {len(hf_wanted)} 个张量,不一致 {len(bad)} 个,索引里有但分片头里没有 {len(absent)} 个",
              bad[:8] + _summarize(absent))

    unused = [k for k in weight_map if k not in hf_wanted]
    unknown = [k for k in unused if not k.startswith(IGNORABLE_CKPT_PREFIXES)]
    check("4. checkpoint 里没用到的键都属于已知可忽略的几类", not unknown,
          f"没用到 {len(unused)} 个,其中不在可忽略清单里的 {len(unknown)} 个", _summarize(unknown))
    if unused:
        print("     没用到的键(按类):")
        for line in _summarize(unused, limit=30):
            print("        " + line)

    print("\n结论:" + ("通过 —— checkpoint 的键和形状与模型对得上。" if ok else "未通过(退出码 1)。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
