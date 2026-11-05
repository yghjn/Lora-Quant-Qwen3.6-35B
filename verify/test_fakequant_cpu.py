#!/usr/bin/env python3
"""fakequant_rtn.py 的端到端自测:造一个合成的 HF checkpoint 再跑全流程。

不需要真权重、不需要 NPU。验的是:
  1. 键名分类正确 —— 路由专家被命中,共享专家/attention/embed/lm_head 被排除
  2. 未分类的键会让脚本**拒绝继续**(负例)
  3. 量化后路由专家确实变了,被排除的张量**逐位未变**
  4. 不含路由专家的分片走复制路径,逐位相同
  5. 非权重文件(config.json / index.json)被带过去
  6. manifest 的字段与 SNR 统计正确
  7. 输出目录非空时拒绝覆盖(负例)
  8. 3bit 的 SNR 落在实测量级

跑法:  python3 -I test_fakequant_cpu.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "src")
SCRIPT = os.path.join(SRC_DIR, "fakequant_rtn.py")
sys.path.insert(0, SRC_DIR)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK ' if cond else 'X  '} {name}{(': ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


# Qwen3.6-35B-A3B 的真实几何,缩小层数与专家数以便 CPU 快跑
D, F, E, L = 2048, 512, 4, 2


def build_ckpt(root: str, *, inject_unknown: bool = False) -> None:
    """造 3 个分片:两个带路由专家(逐专家布局 + 融合布局),一个只有非专家权重。"""
    os.makedirs(root, exist_ok=True)
    torch.manual_seed(7)
    g = lambda *s: (torch.randn(*s) * 0.02).to(torch.bfloat16)  # noqa: E731

    # 分片 1:逐专家布局(HF 常见)
    s1: dict[str, torch.Tensor] = {}
    for i in range(L):
        for e in range(E):
            s1[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = g(F, D)
            s1[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = g(F, D)
            s1[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = g(D, F)
        # 共享专家:**不该被量化**
        s1[f"model.layers.{i}.mlp.shared_expert.gate_proj.weight"] = g(F, D)
        s1[f"model.layers.{i}.mlp.shared_expert.down_proj.weight"] = g(D, F)
        # router gate:不该被量化
        s1[f"model.layers.{i}.mlp.gate.weight"] = g(E, D)
    save_file(s1, os.path.join(root, "model-00001-of-00003.safetensors"))

    # 分片 2:融合布局(3D [E, out, in]),同样该被量化
    s2 = {
        f"model.layers.{L}.mlp.experts.gate_up_proj": g(E, 2 * F, D),
        f"model.layers.{L}.mlp.experts.down_proj": g(E, D, F),
    }
    save_file(s2, os.path.join(root, "model-00002-of-00003.safetensors"))

    # 分片 3:完全没有路由专家 —— 必须走复制路径
    s3: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": g(1024, D),
        "lm_head.weight": g(1024, D),
        "model.norm.weight": g(D),
    }
    for i in range(L):
        s3[f"model.layers.{i}.self_attn.q_proj.weight"] = g(D, D)
        s3[f"model.layers.{i}.self_attn.o_proj.weight"] = g(D, D)
        s3[f"model.layers.{i}.input_layernorm.weight"] = g(D)
        # Gated Delta Net —— 真权重里 40 层有 30 层是这个。放进来是为了让
        # "被排除的张量逐位未变" 那一项也覆盖到它。
        s3[f"model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"] = g(4 * D, D)
        s3[f"model.language_model.layers.{i}.linear_attn.out_proj.weight"] = g(D, 2 * D)
        s3[f"model.language_model.layers.{i}.linear_attn.conv1d.weight"] = g(4 * D, 1, 4)
        s3[f"model.language_model.layers.{i}.linear_attn.A_log"] = g(32)
        s3[f"model.language_model.layers.{i}.linear_attn.dt_bias"] = g(32)
    if inject_unknown:
        # 看着像权重、但不匹配任何模式 —— 脚本必须拒绝继续
        s3["model.layers.0.mystery_block.proj.weight"] = g(D, D)
    save_file(s3, os.path.join(root, "model-00003-of-00003.safetensors"))

    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5_moe", "hidden_size": D, "num_experts": E}, f)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": {}}, f)


def run(args: list[str]) -> subprocess.CompletedProcess:
    # 跟随本测试自己的启动方式。硬写 `-I` 会屏蔽 user site-packages,
    # 而有的环境(本项目的开发机就是)torch 正装在那儿,子进程会 import 不到 torch,
    # 于是每一步都以 ModuleNotFoundError 失败、最后在第 5 步崩在缺文件上 ——
    # 看着像产物没写出来,其实是解释器选错了。
    argv = [sys.executable] + (["-I"] if sys.flags.isolated else []) + [SCRIPT, *args]
    return subprocess.run(
        argv, capture_output=True, text=True, cwd=SRC_DIR
    )


tmp = tempfile.mkdtemp(prefix="fqtest_")
try:
    src = os.path.join(tmp, "src")
    build_ckpt(src)

    # ---------------------------------------------------------- 1 分类
    print("\n[1] 键名分类")
    sys.path.insert(0, SRC_DIR)
    import fakequant_rtn as FQ  # noqa: E402

    cases = {
        "model.layers.0.mlp.experts.3.gate_proj.weight": "routed",
        "model.layers.0.mlp.experts.3.down_proj.weight": "routed",
        "model.layers.9.mlp.experts.gate_up_proj": "routed",
        "model.layers.0.mlp.shared_expert.gate_proj.weight": "excluded",
        "model.layers.0.mlp.gate.weight": "excluded",
        "model.layers.0.self_attn.q_proj.weight": "excluded",
        "model.embed_tokens.weight": "excluded",
        "lm_head.weight": "excluded",
        "model.norm.weight": "excluded",
        "model.layers.0.input_layernorm.weight": "excluded",
        "model.layers.0.mystery_block.proj.weight": "unknown",
        # --- 以下来自 2026-11-05 对 Qwen3.6-35B-A3B 官方权重的实测键名 ---
        # 多模态前缀 model.language_model.* + 融合专家布局
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "routed",
        "model.language_model.layers.0.mlp.experts.down_proj": "routed",
        # Gated Delta Net(混合架构里 40 层有 30 层是它)。这几个当初落进了
        # unknown,因为 self_attn|attention 匹配不到 linear_attn。
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "excluded",
        "model.language_model.layers.0.linear_attn.out_proj.weight": "excluded",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": "excluded",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": "excluded",
        "model.language_model.layers.0.linear_attn.conv1d.weight": "excluded",
        "model.language_model.layers.0.linear_attn.A_log": "excluded",
        "model.language_model.layers.0.linear_attn.dt_bias": "excluded",
        # ViT
        "model.visual.blocks.0.mlp.linear_fc1.weight": "excluded",
        "model.language_model.embed_tokens.weight": "excluded",
    }
    for k, want in cases.items():
        got = FQ.classify(k)
        check(f"{want:<9} {k.split('model.')[-1][:48]}", got == want, f"得到 {got}")

    # ---------------------------------------------------------- 2 负例:未分类
    print("\n[2] 负例:存在未分类的键时必须拒绝")
    src_bad = os.path.join(tmp, "src_bad")
    build_ckpt(src_bad, inject_unknown=True)
    r = run(["--in", src_bad, "--scan"])
    check("scan 返回非 0", r.returncode != 0, f"rc={r.returncode}")
    check("提示了未分类", "未分类" in r.stdout, "")

    # ---------------------------------------------------------- 3 scan 正例
    print("\n[3] 正例:scan 通过")
    r = run(["--in", src, "--scan"])
    check("scan 返回 0", r.returncode == 0, f"rc={r.returncode}\n{r.stdout[-400:]}{r.stderr[-400:]}")
    check("分类无歧义", "分类无歧义" in r.stdout)

    # ---------------------------------------------------------- 4 量化
    print("\n[4] 量化并自带校验  (3bit, group 128)")
    dst = os.path.join(tmp, "dst")
    r = run(["--in", src, "--out", dst, "--bits", "3", "--group", "128"])
    ok = r.returncode == 0
    check("退出码 0", ok, f"rc={r.returncode}")
    if not ok:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
    check("自带校验通过", "校验通过" in r.stdout)

    # ---------------------------------------------------------- 5 逐张量比对
    print("\n[5] 产物逐张量比对")
    n_changed = n_same = 0
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".safetensors"):
            continue
        if not os.path.exists(os.path.join(dst, fn)):
            # 第 4 步已经失败了。这里直接记一条,不要再崩在缺文件的 traceback 上 ——
            # 那个栈会盖掉真正的失败原因。
            FAILED.append(f"产物缺少分片 {fn}(第 4 步没跑成)")
            continue
        with safe_open(os.path.join(src, fn), framework="pt") as fs, safe_open(
            os.path.join(dst, fn), framework="pt"
        ) as fd:
            for k in fs.keys():  # noqa: SIM118
                a, b = fs.get_tensor(k), fd.get_tensor(k)
                if FQ.classify(k) == "routed":
                    if not torch.equal(a, b):
                        n_changed += 1
                    else:
                        FAILED.append(f"路由专家未被量化: {k}")
                elif torch.equal(a, b):
                    n_same += 1
                else:
                    FAILED.append(f"被排除的张量却变了: {k}")
    check("路由专家全部被量化", n_changed > 0, f"{n_changed} 个张量改变")
    check("被排除的张量逐位未变", n_same > 0, f"{n_same} 个张量未变")
    check("无越界改动", not any(f.startswith("被排除") for f in FAILED))

    # ---------------------------------------------------------- 6 复制路径
    print("\n[6] 无目标分片走复制 / 非权重文件带过去")
    import hashlib

    def md5(p: str) -> str:
        return hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: S324

    f3 = "model-00003-of-00003.safetensors"
    check("纯非专家分片逐字节相同", md5(os.path.join(src, f3)) == md5(os.path.join(dst, f3)))
    for fn in ("config.json", "model.safetensors.index.json"):
        check(f"{fn} 被带过去", os.path.exists(os.path.join(dst, fn)))

    # ---------------------------------------------------------- 7 manifest
    print("\n[7] manifest")
    mp = os.path.join(dst, "fakequant_manifest.json")
    check("manifest 存在", os.path.exists(mp))
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        check("记录了 bits", m.get("bits") == 3, str(m.get("bits")))
        check("记录了范围", m.get("scope") == "routed experts only", str(m.get("scope")))
        check("张量数与改变数一致", m.get("n_quantized") == n_changed, f"{m.get('n_quantized')} vs {n_changed}")
        p50 = m["snr_db"]["p50"]
        print(f"      SNR min {m['snr_db']['min']:.2f} / p50 {p50:.2f} / max {m['snr_db']['max']:.2f} dB")
        check("3bit SNR 在实测量级", 8.0 < p50 < 16.0, f"p50={p50:.2f} dB(单测 11.26)")

    # ---------------------------------------------------------- 8 负例:覆盖
    print("\n[8] 负例:输出目录非空时拒绝")
    r = run(["--in", src, "--out", dst, "--bits", "3"])
    check("拒绝覆盖非空目录", r.returncode != 0 and "拒绝覆盖" in (r.stdout + r.stderr))

    # ---------------------------------------------------------- 9 2bit 警告
    print("\n[9] 2bit 会给出警告")
    dst2 = os.path.join(tmp, "dst2")
    r = run(["--in", src, "--out", dst2, "--bits", "2"])
    check("2bit 打了警告", "offline_ptq" in r.stdout, "")

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 62)
if FAILED:
    print(f"失败 {len(FAILED)} 项:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过。fakequant_rtn.py 端到端可用。")
sys.exit(0)
