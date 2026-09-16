# -*- coding: utf-8 -*-
"""下载本地语义编码器（ONNX int8），用于真正的语义检索实测。

为什么必须本地：走外部 embedding API 会把制度原文发往第三方，
与「检索前权限 mask / 数据不出域」的核心卖点直接冲突。本地 int8 模型
只需 onnxruntime + tokenizers，**不需要 torch**，可随交付包进内网离线运行。

⚠️ 诚实声明：本脚本里的候选仓库路径**未能在开发环境验证**
   （该环境无法访问 huggingface.co）。脚本因此会逐个尝试候选、
   报告每个 URL 的实际 HTTP 状态，并在下载后用 tokenizers + onnxruntime
   **真实加载一次**来确认可用——以实际加载结果为准，不以 URL 写得对为准。

用法：
    python fetch_embedding_model.py                      # 用默认端点
    python fetch_embedding_model.py --endpoint https://hf-mirror.com   # 国内镜像
    set HF_ENDPOINT=https://hf-mirror.com && python fetch_embedding_model.py

下载完成后它会打印需要设置的环境变量。
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "models" / "bge-small-zh-v1.5-int8"

# (仓库, 模型文件相对路径) —— 逐个尝试，第一个成功的即采用
CANDIDATES: list[tuple[str, str]] = [
    ("BAAI/bge-small-zh-v1.5", "onnx/model_quantized.onnx"),
    ("BAAI/bge-small-zh-v1.5", "onnx/model.onnx"),
    ("Xenova/bge-small-zh-v1.5", "onnx/model_quantized.onnx"),
    ("onnx-community/bge-small-zh-v1.5-ONNX", "onnx/model_quantized.onnx"),
]
TOKENIZER_CANDIDATES: list[str] = ["tokenizer.json"]


def fetch(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    """下载到 dest。返回 (成功?, 说明)。已存在且非空则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"已存在，跳过（{dest.stat().st_size / 1e6:.1f} MB）"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            total = 0
            with tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
        tmp.replace(dest)
        return True, f"下载完成（{total / 1e6:.1f} MB）"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:                             # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT",
                                                             "https://huggingface.co"),
                        help="HF 端点（国内可用 https://hf-mirror.com）")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.dir)
    endpoint = args.endpoint.rstrip("/")
    print(f"\n端点：{endpoint}")
    print(f"目标目录：{out_dir}\n")

    # 1) tokenizer
    tok_ok = False
    for name in TOKENIZER_CANDIDATES:
        for repo in dict.fromkeys(r for r, _ in CANDIDATES):
            url = f"{endpoint}/{repo}/resolve/main/{name}"
            ok, note = fetch(url, out_dir / name)
            print(f"  [tokenizer] {repo}/{name} → {note}")
            if ok:
                tok_ok = True
                break
        if tok_ok:
            break

    # 2) 模型（逐个候选）
    model_ok = False
    for repo, rel in CANDIDATES:
        fname = Path(rel).name
        url = f"{endpoint}/{repo}/resolve/main/{rel}"
        ok, note = fetch(url, out_dir / fname)
        print(f"  [model] {repo}/{rel} → {note}")
        if ok:
            model_ok = True
            break

    if not (tok_ok and model_ok):
        print("\n❌ 下载未完成。可能原因：网络不可达 / 仓库路径已变 / 需要镜像端点。")
        print("   建议依次尝试：")
        print("     1) 换镜像端点：python fetch_embedding_model.py --endpoint https://hf-mirror.com")
        print("     2) 手动从 HF 仓库页面的 Files 列表下载 tokenizer.json 与 *.onnx，")
        print(f"        放进 {out_dir}（编码器会自动识别 model_quantized/model_int8/model.onnx）")
        return 1

    # 3) 真实加载验证（以能否加载为准，而不是以 URL 写得对为准）
    print("\n校验：尝试用 tokenizers + onnxruntime 真实加载一次…")
    try:
        from cformer_v63.embedding import OnnxEncoder
    except Exception as exc:                             # noqa: BLE001
        print(f"❌ 导入失败：{exc}")
        print("   请先安装：python -m pip install -r requirements-ann.txt")
        return 1

    try:
        encoder = OnnxEncoder(str(out_dir))
        vecs = encoder.encode(["旷工几天会被劝退", "出差需要报备吗"], is_query=True)
        print(f"✅ 加载成功：{encoder.name}，dim={encoder.dim}，"
              f"输出形状 {tuple(vecs.shape)}")
    except Exception as exc:                             # noqa: BLE001
        print(f"❌ 加载失败：{type(exc).__name__}: {exc}")
        print("   常见原因：onnxruntime/tokenizers 未安装；或 tokenizer.json 与模型不配套。")
        return 1

    print("\n下一步（PowerShell）：")
    print(f'  $env:GOVLAYER_ONNX_MODEL="{out_dir}"')
    print("  python eval_semantic_retrieval.py --compare    # 措辞贴近 vs 措辞远离 的语义对比")
    print("  python eval_ann_index.py --sizes 10000 50000 --cluster-multiplier 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
