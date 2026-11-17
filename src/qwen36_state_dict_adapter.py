"""Qwen3.6 的 HF 权重 <-> torchtitan 参数的映射 —— 只改路由专家那两个张量的排布。

为什么需要这个文件
------------------
upstream 的 `Qwen35StateDictAdapter`(`torchtitan/models/qwen3_5/state_dict_adapter.py`,
086bf6c16)对路由专家权重的假设是::

    gate_up_proj  [E, dim, 2*hidden]      down_proj  [E, hidden, dim]

也就是 `x @ W` 的排布,和 transformers 里更早一代的 `Qwen3VLMoeTextExperts` 一样。
而 **Qwen3.6-35B-A3B 真文件里**(2026-11-16 读 26 个分片的文件头核对)是::

    gate_up_proj  (256, 1024, 2048) = [E, 2*hidden, dim]
    down_proj     (256, 2048,  512) = [E, dim, hidden]

即 `nn.Linear` 的 [out, in] 排布,正好是 upstream 假设的转置。40 层 × 2 = 80 个张量,
占全模型参数量的绝大部分。用 upstream 的适配器直接加载,形状对不上。

排布的语义以 vLLM 的 `model_executor/models/qwen3_5.py::load_weights` 为准 ——
基线评测就是用它加载这份 checkpoint 的::

    # qwen3.5 no need to transpose
    # loaded_weight = loaded_weight.transpose(-1, -2)
    if "experts.gate_up_proj" in name:
        loaded_weight = loaded_weight.chunk(2, dim=-2)     # [0] -> w1, [1] -> w3

所以:

* `down_proj` 不转置,直接就是 torchtitan 的 `w2_EDF`;
* `gate_up_proj` 沿 dim -2 对半切,**前一半是 gate(= `w1_EFD`,过 silu 的那一路),
  后一半是 up(= `w3_EFD`)**。torchtitan 的 `GroupedExperts.forward` 是
  `silu(x @ w1^T) * (x @ w3^T)`,和 `act(gate) * up` 对上。

做法
----
不重写 upstream 那两百多行映射。upstream 的结果和真实排布之间只差这两类键各一次
转置,所以在它外面套一层:

* `to_hf`:upstream 给出 `[E, D, 2F]` / `[E, F, D]`,这里转置成 `[E, 2F, D]` / `[E, D, F]`。
  `cat([w1^T, w3^T], dim=-1)` 再转置 == `cat([w1, w3], dim=-2)`,前半 gate 后半 up;
* `from_hf`:先把这两类键转置回 upstream 认识的排布,再交给 upstream。

其余所有键(attention、GDN、共享专家、norm、视觉塔)原样走 upstream。

`verify/probe_weights_map_cpu.py` 用真 checkpoint 的索引和分片文件头核对整张映射:
键是否齐、形状是否一致。**起训练前先跑它**,十几秒,不读权重数据。
"""

from __future__ import annotations

import re
from typing import Any

from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

__all__ = ["Qwen36StateDictAdapter", "is_fused_expert_key"]

_FUSED_EXPERT_KEY = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"
)


def is_fused_expert_key(hf_key: str) -> bool:
    """HF 键是不是路由专家的 `gate_up_proj` / `down_proj`(三维、按专家堆叠的那两个)。"""
    return _FUSED_EXPERT_KEY.match(hf_key) is not None


class Qwen36StateDictAdapter(Qwen35StateDictAdapter):
    """`Qwen35StateDictAdapter`,但路由专家权重按 Qwen3.6 真文件的 [out, in] 排布。"""

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = super().to_hf(state_dict)
        for key in list(hf_state_dict):
            if is_fused_expert_key(key):
                hf_state_dict[key] = hf_state_dict[key].transpose(-2, -1)
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # 不就地改传进来的 dict:dcp_load 之后还会用它。
        upstream_layout = {
            key: (value.transpose(-2, -1) if is_fused_expert_key(key) else value)
            for key, value in hf_state_dict.items()
        }
        return super().from_hf(upstream_layout)
