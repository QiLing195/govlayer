# -*- coding: utf-8 -*-
"""可插拔文本编码器（#4 规模化检索的向量来源）。

两条路线，都是**本地推理、数据不出内网**（这是选型底线，见下）：

  1. OnnxEncoder —— 推荐：ONNX int8 量化小模型（如 BAAI/bge-small-zh-v1.5）。
     只需 onnxruntime + tokenizers，**不需要 torch**，模型约 25–100MB，
     可随交付包一起进内网、离线运行。
  2. HashingEncoder —— 零依赖兜底：字符 n-gram 稳定哈希。
     ⚠️ 它**没有任何语义能力**（"旷工"和"缺勤"对它完全无关），
        只用来验证索引结构与权限分区的正确性。
        用 HashingEncoder 测出来的"召回率"衡量的是**索引相对暴力检索的忠实度**，
        不是语义检索质量——报告里必须分开写，不能混为一谈。

**为什么不走外部 embedding API**：那会把制度原文发往第三方，与
「检索前权限 mask / 数据不出域」这一核心卖点直接冲突。省事，但等于自毁卖点。

**为什么不用 torch 本地模型**：服务镜像目前不含 torch（"镜像小、启动秒级、
可内网离线"全部依据于此），塞进去从 ~150MB 涨到 2GB+。ONNX int8 是同样的本地性、
小得多的体积。

模型目录约定（由 GOVLAYER_ONNX_MODEL 指定）：
    <dir>/tokenizer.json
    <dir>/model_quantized.onnx   （或 model_int8.onnx / model.onnx）
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata

import numpy as np

ONNX_MODEL_ENV = "GOVLAYER_ONNX_MODEL"
# bge-zh 系列官方建议：查询侧加指令前缀，文档侧不加（非对称检索）
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def normalize_text(text: str) -> str:
    """全角→半角、去空白、小写、NFKC —— 与 precise_match 的归一化思路保持一致。"""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", "", text)


# ------------------------------------------------------------------ 零依赖兜底

class HashingEncoder:
    """字符 n-gram 稳定哈希编码器（**无语义能力，仅结构占位**）。

    必须用 blake2b 这类**稳定哈希**，不能用 Python 内置 hash()：
    内置 hash 每个进程加随机盐，索引一旦落盘、重启后向量全部对不上，
    而且这种 bug 只在"重启后"才暴露，非常难查。
    """

    name = "hashing-blake2b（无语义·结构占位）"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)
        self._weights = {1: 1.0, 2: 1.5, 3: 1.0}

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        chars = normalize_text(text)
        for n, weight in self._weights.items():
            for i in range(len(chars) - n + 1):
                gram = chars[i:i + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else vec

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray([self._one(t) for t in texts], dtype=np.float32)


# ------------------------------------------------------------------ 推荐路线

class OnnxEncoder:
    """ONNX int8 本地小模型编码器（数据不出内网，无需 torch）。"""

    def __init__(self, model_dir: str, max_length: int = 512, query_prefix: str = BGE_QUERY_PREFIX):
        import onnxruntime as ort                      # 延迟导入：没装也能跑兜底
        from tokenizers import Tokenizer

        self.model_dir = model_dir
        self.max_length = max_length
        self.query_prefix = query_prefix

        model_file = self._find_model(model_dir)
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"缺少 tokenizer.json：{tokenizer_file}")

        self.session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()

        inputs = {i.name for i in self.session.get_inputs()}
        self._wants_token_type = "token_type_ids" in inputs
        # 不同来源的 ONNX 导出，输出名/输出个数不一样：
        #   · 原始导出通常只有 last_hidden_state
        #   · 有的仓库额外带 pooled 输出，或把句子向量放在索引 0
        # 优先取 last_hidden_state；若只给了池化后的 2D 向量也能直接用。
        out_names = [o.name for o in self.session.get_outputs()]
        self._output_name = ("last_hidden_state" if "last_hidden_state" in out_names
                             else out_names[0])
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self.name = f"onnx-int8:{os.path.basename(model_dir)}"

    @staticmethod
    def _find_model(model_dir: str) -> str:
        for candidate in ("model_quantized.onnx", "model_int8.onnx", "model.onnx"):
            path = os.path.join(model_dir, candidate)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{model_dir} 下找不到 model_quantized.onnx / model_int8.onnx / model.onnx")

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = [self.query_prefix + t for t in texts] if is_query else list(texts)
        encodings = self.tokenizer.encode_batch(payload)

        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention}
        if self._wants_token_type:
            feed["token_type_ids"] = np.asarray(
                [e.type_ids for e in encodings], dtype=np.int64)

        hidden = self.session.run([self._output_name], feed)[0]
        if hidden.ndim == 2:
            # 模型已经做了池化（直接给句子向量）
            pooled = hidden.astype(np.float32)
        else:
            mask = attention[..., None].astype(np.float32)
            pooled = ((hidden * mask).sum(axis=1)
                      / np.maximum(mask.sum(axis=1), 1e-9)).astype(np.float32)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)


# ------------------------------------------------------------------ 工厂

# 编码器缓存：多个知识库/多个调用方会重复请求同一编码器，
# 不缓存就会把 24MB 的 ONNX 模型重复加载 N 次（内存与启动时间都白涨）。
_ENCODER_CACHE: dict[tuple, object] = {}


def build_encoder(prefer: str = "auto", dim: int = 256, verbose: bool = True):
    """按可用性选编码器（同一配置只加载一次）。

    prefer: "auto" | "onnx" | "hashing"
    拿不到 ONNX 模型时**明确降级并说清后果**，绝不静默用无语义编码器冒充语义检索。
    """
    model_dir = os.environ.get(ONNX_MODEL_ENV, "").strip()
    key = (prefer, model_dir, dim)
    cached = _ENCODER_CACHE.get(key)
    if cached is not None:
        return cached

    encoder = _build_encoder(prefer, dim, verbose, model_dir)
    _ENCODER_CACHE[key] = encoder
    return encoder


def _build_encoder(prefer: str, dim: int, verbose: bool, model_dir: str):
    if prefer in ("auto", "onnx") and model_dir:
        try:
            encoder = OnnxEncoder(model_dir)
            if verbose:
                print(f"[encoder] 使用 ONNX int8 本地模型：{encoder.name}（dim={encoder.dim}）")
            return encoder
        except Exception as exc:                       # noqa: BLE001
            if prefer == "onnx":
                raise
            if verbose:
                print(f"[encoder] ⚠️ ONNX 模型不可用（{type(exc).__name__}: {exc}）→ 降级为哈希编码器")

    if prefer == "onnx" and not model_dir:
        raise RuntimeError(f"prefer=onnx 但未设置 {ONNX_MODEL_ENV}")

    if verbose:
        print(f"[encoder] ⚠️ 使用 {HashingEncoder(dim=dim).name}：")
        print("           它没有语义能力（'旷工'与'缺勤'对它无关），只能验证索引结构与权限分区。")
        print("           要测真实语义召回，请设置 GOVLAYER_ONNX_MODEL 指向 int8 模型目录。")
    return HashingEncoder(dim=dim)
