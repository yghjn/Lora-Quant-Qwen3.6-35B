"""周期性保存 LoRA adapter(工单 §6.5)。本仓库自带,不依赖 torchtitan-npu 里的任何文件。

为什么自带
----------
`qwen36_lora_recipe.py` 原先 import 的是
`torchtitan_npu.patches.torchtitan.models.common.adapter_save`。那个文件只存在于
同事 `l84414662` 的工作区里,**是未跟踪文件,从来没提交过** —— 一份干净的
torchtitan-npu clone 里没有它。后果是训练能正常启动、加载完 70 GB 权重、建好
优化器,然后在 `post_optimizer_build_fn` 里 `ModuleNotFoundError`。
(2026-11-16 逐条核对本仓库的 import 时发现;闸门 0.1 走不到这一步,不会报。)

落盘格式和那个文件**保持一致**,因为 `src/merge_lora.py` 是照它读的:

* 文件名 `adapter_r<rank:02d>_step<step:06d>.safetensors`,每个 rank 写自己那片,
  不做 all-gather(专家 adapter 按 EP 分片,rank r 持有自己那段专家);
* 张量只收名字里带 `lora_` 的参数,转 bf16;键是**规范名字**(去掉
  `_checkpoint_wrapped_module` 这类包装模块引入的段),和 state_dict 一致;
* metadata:`step`、`rank`,以及每个 DTensor 参数的
  `<name>.global_shape`(逗号分隔)和 `<name>.placements`。

环境变量:
    ADAPTER_SAVE_DIR     输出目录。不设则不注册,`register_adapter_saver` 返回 False
                         (recipe 里把这个 False 升级成了异常:训完没存等于白训)
    ADAPTER_SAVE_EVERY   每多少步存一次,默认 100
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

__all__ = ["register_adapter_saver", "save_adapters", "shard_name", "canonical_name", "WRAPPER_SEGMENTS"]

# 包装模块插进参数名里的那一段。activation checkpoint 给每一层套了 CheckpointWrapper,
# `named_parameters()` 给出的是
#   layers.0._checkpoint_wrapped_module.moe.routed_experts.inner_experts.w13_lora_a
# 而 state_dict / HF checkpoint 那边的规范名字没有中间那一段。2026-11-17 第一批真
# 训练出来的分片带着它,merge_lora.py 认不出来。落盘时去掉,和 merge_lora.py 里的
# 清单保持一致(verify/test_adapter_save_cpu.py 对拍两边)。
WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod")


def canonical_name(name: str) -> str:
    """去掉参数名里由包装模块引入的那几段。"""
    return ".".join(seg for seg in name.split(".") if seg not in WRAPPER_SEGMENTS)


def shard_name(rank: int, step: int) -> str:
    return f"adapter_r{rank:02d}_step{step:06d}.safetensors"


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _local(p: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """(本 rank 持有的那片, 是否 DTensor)。"""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # 极老的 torch
        return p, False
    if isinstance(p, DTensor):
        return p.to_local(), True
    return p, False


@torch.no_grad()
def save_adapters(model_parts, out_dir: str, step: int) -> int:
    """把本 rank 的 lora_* 参数写成一个 safetensors。返回写了多少个张量。"""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    rank = _rank()
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {"step": str(step), "rank": str(rank)}
    for part in model_parts:
        for name, p in part.named_parameters():
            if "lora_" not in name:
                continue
            key = canonical_name(name)
            if key in tensors:
                raise RuntimeError(f"adapter 参数名规范化之后撞车:{name} -> {key}")
            local, is_dtensor = _local(p)
            tensors[key] = local.detach().to(torch.bfloat16).cpu().contiguous()
            if is_dtensor:
                meta[key + ".global_shape"] = ",".join(str(x) for x in p.shape)
                meta[key + ".placements"] = str(p.placements)
    if not tensors:
        return 0
    path = os.path.join(out_dir, shard_name(rank, step))
    save_file(tensors, path, metadata=meta)
    if rank == 0:
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        logger.info(
            "adapter 已保存: step %d, %d 个张量, 本 rank %.1f MiB -> %s",
            step, len(tensors), nbytes / 2**20, path,
        )
    return len(tensors)


def register_adapter_saver(optimizers, model_parts, total_steps=None) -> bool:
    """在优化器的 post-step 钩子上挂周期保存。返回是否挂上了。

    只挂**第一个**支持 `register_step_post_hook` 的优化器:每个优化器的 step 都会
    触发钩子,挂多个会让计数变成真实步数的倍数。

    ``total_steps``:训练的总步数(整数,或一个调用时才求值的函数)。给了的话,
    **最后一步一定保存**,不管它是不是 ``ADAPTER_SAVE_EVERY`` 的倍数。
    不给就只按周期存 —— 那样 256 步配默认的每 100 步,最后一次保存停在第 200 步,
    后 56 步白训,而且没有任何提示(2026-11-19 整理流程时发现,还没人踩到)。
    """
    out_dir = os.environ.get("ADAPTER_SAVE_DIR")
    if not out_dir:
        logger.warning("ADAPTER_SAVE_DIR 未设置 —— adapter 不会被保存")
        return False
    every = int(os.environ.get("ADAPTER_SAVE_EVERY", "100"))
    if every <= 0:
        raise ValueError(f"ADAPTER_SAVE_EVERY 必须是正整数,得到 {every}")
    state = {"step": 0}

    def _total() -> int:
        try:
            value = total_steps() if callable(total_steps) else total_steps
            return int(value) if value else 0
        except Exception:  # noqa: BLE001 - 求不出总步数就退回只按周期存
            return 0

    def _hook(*_args, **_kwargs):
        state["step"] += 1
        if state["step"] % every == 0 or state["step"] == _total():
            save_adapters(model_parts, out_dir, state["step"])

    opts = list(optimizers)
    for opt in opts:
        if hasattr(opt, "register_step_post_hook"):
            opt.register_step_post_hook(_hook)
            total = _total()
            logger.info(
                "adapter 保存已注册: 每 %d 步%s -> %s(挂在 %s 上;共 %d 个优化器,只挂一个)",
                every, f",以及最后一步(第 {total} 步)" if total else "(不知道总步数,最后一步不保证保存)",
                out_dir, type(opt).__name__, len(opts),
            )
            return True
    logger.warning("没有优化器支持 register_step_post_hook —— adapter 不会被保存")
    return False
