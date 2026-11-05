"""Qwen3.6-35B-A3B:低位宽基座 + 分组专家 LoRA 的 trainer 配置(工单阶段 6)。

用法 —— 把本仓库的 ``src/`` 放上 PYTHONPATH,然后::

    export PYTHONPATH=<repo>/src:$PYTHONPATH
    NGPU=8 MODULE=qwen36_lora_recipe CONFIG=qwen36_35b_a3b_lora_smoke \\
      HF_ASSETS_PATH=<tokenizer 目录> \\
      DATA_FILES=<prepare_corpus.py 的产物.jsonl> \\
      ADAPTER_SAVE_DIR=<adapter 输出目录> ADAPTER_SAVE_EVERY=20 \\
      bash scripts/run_train.sh --training.steps 20

本模块把 ``torchtitan_npu.models.qwen3_5`` 的全部公开名字搬进自己的命名空间,
所以它可以整体替代那个 MODULE,**不用改 torchtitan-npu 仓库的任何一个文件**
(那个 clone 里有别人未提交的工作,见 README 的"不要动的东西")。

退路:如果启动器坚持 MODULE 必须是 ``torchtitan_npu.models.qwen3_5``,
就在该包 ``config_registry.py`` 末尾加一行::

    from qwen36_lora_recipe import *  # noqa: F401,F403

----

**为什么能复用 DSv4 的 LoRA converter —— 已核对,不是假设。**

* ``torchtitan_npu/models/deepseek_v4/lora.py:487`` 的匹配后缀是
  ``moe.routed_experts.inner_experts``,靠配置树 FQN 后缀匹配(不是 isinstance)。
* ``RoutedExperts`` 和它的 ``inner_experts`` 字段定义在
  ``third_party/torchtitan/torchtitan/models/common/moe.py:129``,是 DSv4 和
  Qwen3.5 **共用**的;Qwen3.5 的解码层把它挂在 ``self.moe``
  (``models/qwen3_5/model.py:617``)。后缀逐段对上。
* ``grouped_lora_class(parent_cls)`` 是以父类为参数的工厂,只读
  ``w1_EFD`` / ``w2_EDF`` / ``w3_EFD`` 和 ``swiglu_limit``
  (``getattr(..., 0.0)``,为 0 就跳过),不含任何 DeepSeek 专有几何。

仍然要跑 ``verify/probe_lora_tree.py``(工单 §0.1)在纯 CPU 上确认转换后的
config 类型真的变成了 ``GroupedLoRAExperts.Config`` —— 静态读码只能说明
路径对得上,说明不了 ``convert()`` 走完之后树长什么样。
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

logger = logging.getLogger(__name__)

# 搬入 NPU 侧 qwen3_5 的命名空间。import 本身也触发该包的注册副作用,
# 所以 MODULE=qwen36_lora_recipe 等价于 MODULE=torchtitan_npu.models.qwen3_5 + 本文件。
from torchtitan_npu.models import qwen3_5 as _npu_qwen35  # noqa: E402
from torchtitan_npu.models.qwen3_5 import config_registry as _npu_registry  # noqa: E402

for _n, _v in vars(_npu_registry).items():
    if not _n.startswith("__") and _n not in globals():
        globals()[_n] = _v
for _n, _v in vars(_npu_qwen35).items():
    if not _n.startswith("__") and _n not in globals():
        globals()[_n] = _v
del _n, _v

from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw  # noqa: E402
from torchtitan.hf_datasets.text_datasets import ChatDataLoader  # noqa: E402
from torchtitan.models.qwen3_5 import model_registry  # noqa: E402
from torchtitan.models.qwen3_5.config_registry import qwen35_35b_a3b  # noqa: E402
from torchtitan.trainer import Trainer  # noqa: E402

from torchtitan_npu.models.qwen3_5.config_registry import (  # noqa: E402
    _messages,
    _parallelize_text,
)

# 工单 §6.1 的默认值。rank 16 的 adapter 是 0.92 G 参数 / 1.85 GiB bf16,
# 只有 DSv4.1 rank16 的 21.9%(工单 §6 的 rank 表)。
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32.0


def _lora_converters(rank: int, alpha: float) -> list:
    """只给路由专家加 adapter 的 converter 链(工单 §6.1)。

    **没有 int8 冻结基座,这条链只有一环。** DSv4.1 需要 int8 是因为 543.6 B
    路由专家 bf16 按 16 片摊 = 68.1 GiB/die 超过 61.27 GiB 容量;Qwen3.6 的
    路由专家是 32.2 G 参数,bf16 全模型约 70 GB,8 卡摊下来 8.75 GB/卡,
    bf16 直接装得下(工单 §2.1)。所以也就不存在 DSv4.1 那个
    "Int8FrozenBaseConverter 排后面就静默变成 bf16 基座"的顺序坑。

    ``target_modules=()`` 即不给 dense 层加 adapter。理由和 DSv4.1 一样:
    量化只做了路由专家(工单 §1.5),attention / router 没被动过,
    给它们加 adapter 等于训出一个部署时用不上的模型 —— DSv4.1 实测 936 个
    adapter 键里 736 个是这类白训的。

    ``strict=True`` 是**要的**:一个分组专家都没匹配到时
    ``_validate_matches``(lora.py:582)会抛。没有它,后缀对不上就会静默
    退化成一个没有 adapter 的全量微调。
    """
    from torchtitan_npu.models.deepseek_v4.lora import DeepSeekV4LoRAConverter

    return [
        DeepSeekV4LoRAConverter.Config(
            rank=rank,
            alpha=alpha,
            rank_experts=rank,
            target_modules=(),
            adapt_routed_experts=True,
            strict=True,
        )
    ]


def _freeze_non_adapter(model) -> int:
    """冻结所有非 adapter 参数(工单 §6.2)。

    抄 ``deepseek_v4_1/parallelize.py:238-249``。判据是"模型里有没有名字含
    ``lora_`` 的参数",**不是** ``getattr(model, "lora_config")`` ——
    converter 换的是配置树,模型对象上不保证留下那个属性,用它会静默不冻结,
    于是 32 G 基座连着优化器状态一起上,显存账立刻不成立。
    """
    has_lora = any("lora_" in n for n, _ in model.named_parameters())
    if not has_lora:
        raise RuntimeError(
            "模型里没有任何名字含 lora_ 的参数 —— converter 没生效。"
            "先跑 verify/probe_lora_tree.py(工单 §0.1)看配置树到底变成了什么。"
        )
    frozen = 0
    trainable = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            trainable += 1
        else:
            p.requires_grad_(False)
            frozen += 1
    logger.info("LoRA: 冻结了 %d 个非 adapter 参数,可训 adapter 张量 %d 个", frozen, trainable)
    return frozen


def _parallelize_lora(model, **kwargs):
    """``_parallelize_text`` + 冻结。

    冻结必须在 parallelize **之后**:converter 是在 model_registry 里跑的,
    参数要等模型实例化出来才存在。
    """
    out = _parallelize_text(model, **kwargs)
    _freeze_non_adapter(out if out is not None else model)
    return out


def _lora_model_spec(rank: int, alpha: float):
    """带 LoRA converter 的 ModelSpec。

    converter **不是** ``Trainer.Config`` 的字段,而是 ``model_registry`` 的
    参数(``models/qwen3_5/__init__.py:1100``):它在那里对 config 树逐环
    ``convert()``。所以不能拿 ``qwen35_35b_a3b()`` 的 spec 回头改,
    必须重新 registry 一次。
    """
    spec = model_registry(
        "35B-A3B",
        moe_comm_backend="standard",
        converters=_lora_converters(rank, alpha),
    )
    return replace(spec, parallelize_fn=_parallelize_lora)


def _wire_adapter_saver(config: Trainer.Config) -> Trainer.Config:
    """注册 adapter 落盘钩子(工单 §6.5)。

    必须挂在 ``post_optimizer_build_fn`` 上:它在 init_states 之后,而
    ``register_adapter_saver`` 要拿到已经建好的 optimizers。链式包装 ——
    上游这个钩子原本是 ``register_moe_load_balancing_hook``,不能丢。

    **和 DSv4.1 的区别**:那边把 ``register_adapter_saver`` 放在
    ``UNISVQ_CKPT`` 分支内部(config_registry.py:369),所以不设那个环境变量
    adapter 根本不会保存。我们没有 int8 加载器这一步,所以无条件注册,
    并且把"没设 ADAPTER_SAVE_DIR"从 warning 升级成异常 ——
    "训完了但没存"是会白跑几小时的失败,不值得留给一条 warning。
    """
    spec = config.model_spec
    prev = getattr(spec, "post_optimizer_build_fn", None)

    def _hook(optimizers, model_parts, parallel_dims):
        if prev is not None:
            prev(optimizers, model_parts, parallel_dims)
        from torchtitan_npu.patches.torchtitan.models.common.adapter_save import (
            register_adapter_saver,
        )

        if not register_adapter_saver(optimizers, model_parts):
            raise RuntimeError(
                "ADAPTER_SAVE_DIR 未设置 —— adapter 不会被保存,等于白训。"
                "设它指向一个输出目录后重跑。"
            )

    config.model_spec = replace(spec, post_optimizer_build_fn=_hook)
    return config


def _base_lora_config(rank: int, alpha: float, steps: int) -> Trainer.Config:
    config = qwen35_35b_a3b()

    # loss 里的 global_vocab_size 是用未转换的 spec 算的。LoRA converter
    # target_modules=() 只动路由专家,碰不到 embedding / lm_head,词表不变,
    # 所以沿用上游算好的那个值是对的。
    config.model_spec = _lora_model_spec(rank, alpha)
    config = _wire_adapter_saver(config)

    # 语料:prepare_corpus.py 产出的 jsonl,每行一个 {"messages": [...]}。
    config.dataloader = ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": os.environ["DATA_FILES"], "split": "train"},
        sample_processor=_messages,
        infinite=True,
    )

    # 工单 §6.3:AdamW。weight_decay=0 —— adapter 学的是对一份**具体**量化误差的
    # 补偿,把它往 0 拉没有正则意义,只是把学到的东西扣掉一点。
    # lr 1e-4 跟齐 qwen35_27b_4k_sft;上游 35b_a3b 的 5e-4 是预训练的量级。
    config.optimizer = default_adamw(lr=1e-4, weight_decay=0.0)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=max(1, min(16, steps // 10)), decay_ratio=1.0, min_lr_factor=0.1
    )

    # **并行度必须改。** 上游 qwen35_35b_a3b 是 tp=2 / ep=8 = 16 rank,
    # 我们只有 8 张卡。tp=1 / ep=8,dp_shard=-1 推出来是 1。
    # 256 个专家摊 8 卡 = 32 个/卡。
    config.parallelism = replace(
        config.parallelism,
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        expert_parallel_degree=8,
    )

    config.training = replace(
        config.training,
        local_batch_size=1,
        seq_len=4096,
        steps=steps,
    )
    config.metrics = replace(config.metrics, log_freq=1)

    # 基座不训,别存:全模型 DCP 约 70 GB,而 adapter 走 adapter_save 自己落盘。
    config.checkpoint = replace(
        config.checkpoint, interval=max(steps, 1), last_save_model_only=True
    )
    return config


def qwen36_35b_a3b_lora_smoke() -> Trainer.Config:
    """工单 §1.3:20 步冒烟。看的是 loss 下降 + adapter 分片落盘,不是精度。

    **先跑这个。** 它和交付配置走完全同一条代码路径,只是步数短,
    所以 converter / 冻结 / adapter 落盘三件事它全都验得到。
    """
    config = _base_lora_config(DEFAULT_RANK, DEFAULT_ALPHA, steps=20)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=2, decay_ratio=1.0, min_lr_factor=0.1
    )
    return config


def qwen36_35b_a3b_lora_r16() -> Trainer.Config:
    """交付用:rank 16 / alpha 32(scaling 2),256 步。

    初始权重必须是阶段 1(RTN 3bit)或阶段 2(LDLQ)的**假量化 bf16
    checkpoint** —— adapter 要补偿的就是那一份具体的量化误差,
    载成原始权重训出来的 adapter 放到量化基座上是另一回事(工单 §6.4)。
    启动时确认日志里没有"保持随机初始化"一类告警。
    """
    return _base_lora_config(DEFAULT_RANK, DEFAULT_ALPHA, steps=256)


def qwen36_35b_a3b_lora_r4() -> Trainer.Config:
    """语料少的时候用:rank 4 / alpha 8,**保持 alpha/rank = 2** 与 r16 可比。

    DSv4.1 的教训:rank16 的 adapter 是 4.22 B 可训参数而 trace 语料只有
    1.62 M token,严重欠定,于是他们退到 rank 2。Qwen3.6 单专家 rank4
    约 34 K 参数,按 top-8/256 摊算所需 token 量比 DSv4.1 宽松得多。
    """
    return _base_lora_config(4, 8.0, steps=256)
