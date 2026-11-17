"""Qwen3.6-35B-A3B:低位宽基座 + 分组专家 LoRA 的 trainer 配置(工单阶段 6)。

用法 —— 把本仓库的 ``src/`` 放上 PYTHONPATH,然后::

    export PYTHONPATH=<repo>/src:$PYTHONPATH
    NGPU=8 MODULE=qwen36_lora_recipe CONFIG=qwen36_35b_a3b_lora_smoke \\
      HF_ASSETS_PATH=<HF 格式的 checkpoint 目录,绝对路径> \\
      DATA_FILES=<prepare_corpus.py 的产物.jsonl,绝对路径> \\
      ADAPTER_SAVE_DIR=<adapter 输出目录> ADAPTER_SAVE_EVERY=20 \\
      bash scripts/run_train.sh --training.steps 20

**这四个环境变量是本模块自己读的**(``_wire_weights`` / ``_base_lora_config`` /
``qwen36_adapter_save``),不是 ``scripts/run_train.sh`` 读的 —— 那个脚本只认
``NGPU`` / ``MODULE`` / ``CONFIG``,其余一律要写成 ``--xxx`` 命令行参数。
所以换成别的 MODULE 时,``HF_ASSETS_PATH=...`` 这种写法**不会生效也不会报错**。

``HF_ASSETS_PATH`` 一个目录管三件事(``trainer.py:567/598``、``dcp.py:534``):
tokenizer(``tokenizer.json`` + ``tokenizer_config.json`` + chat template)、
权重索引(``model.safetensors.index.json``)、权重分片本身。原始下载的
Qwen3.6-35B-A3B 目录、``fakequant_rtn.py`` 的输出目录都满足(后者会把非权重文件
原样带过去)。**NGPU 必须是 8**:配置里 expert_parallel_degree=8,
而且 bf16 全模型约 70 GB,一张 60 GB 的卡装不下。

本模块把 ``torchtitan_npu.models.qwen3_5`` 的全部公开名字搬进自己的命名空间,
所以它可以整体替代那个 MODULE,**不用改 torchtitan-npu 仓库的任何一个文件**
(那个 clone 里有别人未提交的工作,见 README 的"不要动的东西")。

**"不用改"在一份干净的 clone 上要靠本仓库另外两个文件才成立**(2026-11-16 补)::

    src/qwen36_lora_compat.py    converter。已提交的 lora.py 把根配置写死成 DSv4 的类,
                                 Qwen 过不去;同事工作区里修了但没提交。
    src/qwen36_adapter_save.py   adapter 落盘。原先 import 的 adapter_save.py 是同事
                                 工作区里的未跟踪文件,干净 clone 里根本没有。

第一次上卡(2026-11-16)又补了两处 upstream 对 Qwen + LoRA 不成立的地方::

    qwen36_lora_compat.ensure_adapter_sharding   adapter 参数的分布式切分声明。
                                 Qwen 的声明表里只有 3 个基座权重,并行化时报
                                 "w13_lora_a has no placement declared"。
    src/qwen36_state_dict_adapter.py   路由专家权重的排布。upstream 假设的是
                                 Qwen3.6 真文件的转置,加载时形状对不上。

本工单最初是照同事的**工作区**核的,不是照已提交的 ``3f9f99a``,这两处就是差出来的。

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

# triton-ascend 第一次跑 GDN kernel 时要现编 C++ 启动代码,在 torch 2.14 + CANN 8.x 上
# 要走本仓库的编译器包装才编得过(见 qwen36_triton_env.py / env/triton_cxx.sh)。
# 放在导入时:每个 rank 的进程都会导入本模块,而 kernel 到第一次前向才编。
from qwen36_triton_env import ensure_triton_compiler as _ensure_triton_compiler  # noqa: E402

_TRITON_COMPILER_NOTE = _ensure_triton_compiler()

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
    bf16 直接装得下(工单 §C.1)。所以也就不存在 DSv4.1 那个
    "Int8FrozenBaseConverter 排后面就静默变成 bf16 基座"的顺序坑。

    ``target_modules=()`` 即不给 dense 层加 adapter。理由和 DSv4.1 一样:
    量化只做了路由专家(工单 §B.5),attention / router 没被动过,
    给它们加 adapter 等于训出一个部署时用不上的模型 —— DSv4.1 实测 936 个
    adapter 键里 736 个是这类白训的。

    ``strict=True`` 是**要的**:一个分组专家都没匹配到时
    ``_validate_matches``(lora.py:582)会抛。没有它,后缀对不上就会静默
    退化成一个没有 adapter 的全量微调。
    """
    # 不直接用 DeepSeekV4LoRAConverter.Config:已提交的 3f9f99a 里它的 convert()
    # 最后一行把根配置派生成写死的 DSv4 根配置类,Qwen 会报
    # "missing 4 required keyword-only arguments: 'hc_head', ..."。
    # qwen36_lora_compat 在需要时套一层,不需要时原样返回原 converter。
    # verify/probe_lora_tree.py 用的是同一个入口。
    from qwen36_lora_compat import describe as _compat_describe
    from qwen36_lora_compat import lora_converter_config

    logger.info("LoRA converter: %s", _compat_describe())
    return [
        lora_converter_config(
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
    _disable_load_balancing(spec.model)

    # Qwen3.6 真文件里路由专家权重是 [out, in] 排布,upstream 的适配器假设的是它的
    # 转置,直接加载形状对不上。见 qwen36_state_dict_adapter.py。
    from qwen36_state_dict_adapter import Qwen36StateDictAdapter

    return replace(
        spec,
        parallelize_fn=_parallelize_lora,
        state_dict_adapter=Qwen36StateDictAdapter,
    )


def _disable_load_balancing(model_config) -> int:
    """关掉 MoE 的负载均衡偏置。返回改了多少层。

    upstream ``MoE.Config.load_balance_coeff`` 默认 ``1e-3``(``models/common/moe.py``):
    每层注册一个 ``expert_bias_E`` buffer(初值 0),由优化器的 pre-hook 每步按专家
    负载调整,并**加到路由打分上**。那是给从头预训练用的。

    我们是冻结基座、只训补偿 adapter:

    * 这个偏置会让训练时的路由一步步偏离部署时的路由(256 步最多漂 0.256),
      adapter 学到的是"偏置后的路由"下的补偿;
    * 它不在 HF checkpoint 里,``merge_lora.py`` 也不会导出它 —— 部署时不存在。

    所以设成 ``None``:不注册 buffer,路由打分就是 checkpoint 里 router 的原值。
    """
    changed = 0
    for layer in model_config.layers:
        moe = getattr(layer, "moe", None)
        if moe is not None and getattr(moe, "load_balance_coeff", None) is not None:
            moe.load_balance_coeff = None
            changed += 1
    if changed:
        logger.info("LoRA: 关掉了 %d 层的 MoE 负载均衡偏置(load_balance_coeff=None)", changed)
    return changed


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
        # 本仓库自带的 saver。原先 import 的
        # torchtitan_npu.patches.torchtitan.models.common.adapter_save 是同事工作区里
        # 的未跟踪文件,干净 clone 里没有 —— 会在加载完权重、建好优化器之后才炸。
        from qwen36_adapter_save import register_adapter_saver

        if not register_adapter_saver(optimizers, model_parts):
            raise RuntimeError(
                "ADAPTER_SAVE_DIR 未设置 —— adapter 不会被保存,等于白训。"
                "设它指向一个输出目录后重跑。"
            )

    config.model_spec = replace(spec, post_optimizer_build_fn=_hook)
    return config


# examples/qwen3_5/*.sh 默认带的那个 override。上游 GatedDeltaKernel 的默认后端是
# "fla_chunked"(FLA 的 CUDA Triton kernel);NPU 上走的是这个 override 换上去的
# Triton-Ascend kernel。配置工厂本身不注入它("不会静默注入 NPU 实现",
# examples/qwen3_5/readme.md),是 examples 的启动脚本用 --override.imports 加的 ——
# 我们直接调 scripts/run_train.sh,没有那层脚本,所以在这里加。
NPU_GDN_OVERRIDE = "torchtitan_npu.override.qwen3_5.gated_delta.npu"


def _wire_weights(config: Trainer.Config) -> Trainer.Config:
    """基座权重从哪儿来(工单 §0.4 / §6.4)。**没有这一步,基座是随机初始化的。**

    2026-11-16 核对 ``components/checkpointer/base.py``(上游 086bf6c16)才发现:

    * ``CheckpointManager.Config.enable`` 默认 ``False``,而 ``load()`` 的第一行是
      ``if not self.enable: return False``。上游 ``qwen35_35b_a3b()`` 没开它,
      本文件原先也只改了 ``interval``。结果是训练正常启动、loss 正常下降、
      ``exit=0``,但 adapter 补偿的是一个**随机基座** —— 和 DSv4.1 不设
      ``UNISVQ_CKPT`` 是同一种白跑,而且一条告警都没有。
    * ``hf_assets_path`` 的上游默认值是 ``./assets/hf/Qwen3.5-35B-A3B``(相对于
      启动目录)。``HF_ASSETS_PATH`` 这个环境变量在 ``scripts/run_train.sh`` 和
      ``3f9f99a`` 的配置里**没有任何人读**,只有 ``examples/`` 下的脚本把它转成
      ``--hf-assets-path``。本工单的命令一直是按"它会被读"写的。

    所以这里:读 ``HF_ASSETS_PATH``(必填),打开 checkpoint,并且
    ``initial_load_in_hf=True`` —— 没给 ``initial_load_path`` 时上游就从
    ``hf_assets_path`` 加载 HF safetensors(``dcp.py:534``)。

    ``load_only=True``:允许加载、禁止一切保存。``enable=True`` 之后最后一步会
    强制存一份全模型(``_should_save`` 的 ``last_step`` 分支),约 70 GB,
    而我们要的只是 adapter,它由 ``qwen36_adapter_save`` 自己落盘。

    ``INITIAL_LOAD_PATH``(可选):权重和 tokenizer 不在同一个目录时才用。
    """
    assets = os.environ.get("HF_ASSETS_PATH", "").strip()
    if not assets:
        raise RuntimeError(
            "HF_ASSETS_PATH 未设置 —— 它是 HF 格式的 checkpoint 目录"
            "(tokenizer + model.safetensors.index.json + 权重分片)。"
            "不设的话基座不会被加载,等于在随机权重上训 adapter。"
        )
    if not os.path.isabs(assets):
        raise RuntimeError(
            f"HF_ASSETS_PATH 必须是绝对路径(现在是 {assets!r})—— "
            "训练是 cd 到 torchtitan-npu 目录里起的,相对路径会指到别处。"
        )
    initial = os.environ.get("INITIAL_LOAD_PATH", "").strip() or None

    config.hf_assets_path = assets
    config.checkpoint = replace(
        config.checkpoint,
        enable=True,
        load_only=True,
        initial_load_path=initial,
        initial_load_in_hf=True,
        initial_load_model_only=True,
    )
    logger.info(
        "基座权重: HF safetensors,来自 %s;tokenizer 来自 %s;checkpoint 只读不存",
        initial or assets,
        assets,
    )
    return config


def _wire_npu_overrides(config: Trainer.Config) -> Trainer.Config:
    """把 NPU 的 GDN kernel override 加进 ``override.imports``(已有就不重复加)。"""
    imports = list(config.override.imports)
    if NPU_GDN_OVERRIDE not in imports:
        imports.append(NPU_GDN_OVERRIDE)
    config.override = replace(config.override, imports=imports)
    return config


def _base_lora_config(rank: int, alpha: float, steps: int) -> Trainer.Config:
    logger.info("Triton 编译器: %s", _TRITON_COMPILER_NOTE)
    config = qwen35_35b_a3b()
    config = _wire_weights(config)
    config = _wire_npu_overrides(config)

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
    # 我们只有 8 张卡。tp=1 / ep=8,dp_shard=-1 推出来是 8
    # (verify/probe_parallelize_cpu.py 实测:world=8 dp_shard=8 tp=1 ep=8)。
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
    # 真正挡住保存的是 _wire_weights 里的 load_only=True;interval 只是兜底。
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
