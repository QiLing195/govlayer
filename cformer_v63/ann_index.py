# -*- coding: utf-8 -*-
"""按权限分区的 ANN 检索索引（#4 规模化检索）。

核心设计：**权限必须在检索前结构性生效，而不是检索后再过滤。**

为什么不能"先跑全局 ANN 检索、再按权限过滤结果"（这是很容易犯的错）：
  1. 索引文件本身包含全部密级内容 → 索引泄漏 = 全库泄漏；
  2. 受限条目会挤占 top-k 名额 → 权限越低召回越差，且差多少不可预测；
  3. 命中数与耗时随密级变化 → 时序/结果侧信道可反推"受限内容里有没有这条"。

因此本索引按「可见该内容所需的最低级别」**物理分区**：
  级别 L 的查询只遍历 level ≤ L 的分区，level > L 的分区在代码路径上**不可达**。
  不泄漏不是"过滤掉"的结果，而是"没有路径能到达"的结果。

分区按 level 而不是按 role 命名：level（0 最低）是跨域通用语义，
role 名各库不同（员工/学生/孩子）。按 level 分区，一个索引可服务多套角色命名，
这也是 server/auth.py 里"令牌 → 级别 → 角色"能用同一套逻辑的原因。

单个对象在多个级别都有内容时（如"考勤处罚"在 0/1/2 都有条款），
会为每个级别各建一条索引项，文本是**该级别下的累计可见内容**
（level ≤ L 的全部条款拼接）——这样级别 L 的命中结果与 GovLayer 实际能给该角色
看到的内容一致，不会出现"检索到但展示为空"的错配。

依赖：numpy（不含 torch —— 保持服务镜像可离线、体积可控）。
编码器可插拔，见 cformer_v63/embedding.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（零向量用 eps 保护，避免除零产生 NaN）。"""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _kmeans(matrix: np.ndarray, k: int, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """朴素 k-means（余弦相似度，输入须已归一化）。返回 (质心, 每个点的簇号)。

    不用 sklearn：多一个依赖不值当，且这里只是 POC 规模的聚类。
    """
    n = len(matrix)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centroids = matrix[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(iters):
        assign = (matrix @ centroids.T).argmax(axis=1)
        counts = np.bincount(assign, minlength=k)
        summed = np.zeros_like(centroids)
        np.add.at(summed, assign, matrix)
        nonempty = counts > 0
        summed[nonempty] /= counts[nonempty, None]
        # 空簇保留原质心：否则该簇变 NaN，后续所有相似度计算被污染
        summed[~nonempty] = centroids[~nonempty]
        centroids = l2_normalize(summed)

    assign = (matrix @ centroids.T).argmax(axis=1)
    return centroids, assign


@dataclass
class _Entry:
    object_id: str
    level: int
    text: str
    vec: np.ndarray | None = None


@dataclass
class _Partition:
    """一个权限级别的子索引：自己的条目、质心、倒排表。彼此完全独立。"""

    level: int
    entries: list[_Entry] = field(default_factory=list)
    vecs: np.ndarray | None = None
    centroids: np.ndarray | None = None
    lists: list[np.ndarray] = field(default_factory=list)

    def finalize(self, dim: int, n_clusters: int | None, iters: int, seed: int) -> None:
        if not self.entries:
            self.vecs = np.zeros((0, dim), dtype=np.float32)
            return
        self.vecs = np.asarray([e.vec for e in self.entries], dtype=np.float32)
        if n_clusters is None:
            self.centroids = None          # 小分区直接暴力扫，聚类反而更慢
            return
        self.centroids, assign = _kmeans(self.vecs, n_clusters, iters, seed + self.level)
        self.lists = [np.flatnonzero(assign == c) for c in range(len(self.centroids))]

    def search(self, query_vec: np.ndarray, n_probe: int, top_k: int,
               min_score: float | None = None) -> tuple[list[tuple[str, float, int]], int]:
        """返回 ([(object_id, score, level)], 实际扫描的向量数)。

        min_score: 相似度下限。**接进产品时必须给**——
        向量检索永远会返回 top-k，哪怕所有条目与问题毫不相关（相似度全为 0）。
        没有下限，"制度空白/范围外"这类判定会被静默废掉：系统对任何问题都答得出来。
        """
        if not self.entries:
            return [], 0
        assert self.vecs is not None

        def _keep(scores: np.ndarray, order: np.ndarray, entry_of: np.ndarray) -> list:
            """order 是 **scores 的位置**下标；entry_of[pos] 才是 **entries 的下标**。

            这两个下标空间必须分开处理。聚类分支里 cand 本身就是 entries 下标数组，
            直接拿它索引 scores（长度只有 len(cand)）会越界或静默取到错误分数——
            实测 n_probe=1 且簇数>1 时必然 IndexError。
            """
            out: list[tuple[str, float, int]] = []
            for pos in order:
                if min_score is not None and scores[pos] < min_score:
                    continue
                out.append((self.entries[entry_of[pos]].object_id,
                            float(scores[pos]), self.level))
                if len(out) >= top_k:      # 过滤之后再截断，否则名额会被不合格项占掉
                    break
            return out

        if self.centroids is None:
            scores = self.vecs @ query_vec
            order = np.argsort(-scores)
            return _keep(scores, order, np.arange(len(self.entries))), len(self.entries)

        centroid_sims = self.centroids @ query_vec
        probe = np.argsort(-centroid_sims)[:max(1, min(n_probe, len(self.centroids)))]
        picked = [self.lists[p] for p in probe if len(self.lists[p])]
        if not picked:
            return [], 0
        cand = np.concatenate(picked)
        scores = self.vecs[cand] @ query_vec
        order = np.argsort(-scores)
        return _keep(scores, order, cand), len(cand)


class PermissionPartitionedIndex:
    """按权限级别分区的检索索引。查询级别 L 永远不触碰 level > L 的分区。"""

    def __init__(self, encoder, n_probe: int = 2, kmeans_iters: int = 10, seed: int = 0,
                 brute_force_threshold: int = 2048, cluster_multiplier: float = 1.0):
        """
        encoder: 需提供 .encode(texts, is_query=False) -> (n, dim) 且已归一化，及 .dim/.name
        n_probe: 每级查询探测的质心数（越大越准越慢）
        brute_force_threshold: 分区条目数低于此值就不聚类，直接精确扫描
        cluster_multiplier: 质心数 = sqrt(条目数) × 该系数（调大可提高召回，代价是内存与构建时间）
        """
        self.encoder = encoder
        self.n_probe = n_probe
        self.kmeans_iters = kmeans_iters
        self.seed = seed
        self.brute_force_threshold = brute_force_threshold
        self.cluster_multiplier = float(cluster_multiplier)
        self.partitions: dict[int, _Partition] = {}
        self.build_seconds = 0.0
        self.last_scanned = 0
        self.last_searched_levels: list[int] = []

    # ---------------- 构建 ----------------
    @staticmethod
    def visible_text_per_level(obj: dict) -> dict[int, str]:
        """算出对象在各级别下的累计可见文本（与 GovLayer 的字段级分级语义一致）。"""
        raw = obj.get("levels") or {}
        levels = {}
        for key, content in raw.items():
            try:
                levels[int(key)] = content
            except (TypeError, ValueError):
                continue
        if not levels:
            return {}

        head = " ".join(filter(None, [str(obj.get("title", "")),
                                      " ".join(obj.get("keywords") or [])]))
        out: dict[int, str] = {}
        acc: list[str] = []
        for lvl in sorted(levels):
            acc.append(str(levels[lvl]))
            out[lvl] = (head + " " + " ".join(acc)).strip()
        return out

    def build(self, objects: list[dict], *, verbose: bool = False) -> "PermissionPartitionedIndex":
        import time

        t0 = time.perf_counter()
        by_level: dict[int, list[_Entry]] = {}
        for obj in objects:
            oid = str(obj.get("id", ""))
            if not oid:
                continue
            for lvl, text in self.visible_text_per_level(obj).items():
                by_level.setdefault(lvl, []).append(_Entry(oid, lvl, text))

        for lvl, entries in by_level.items():
            vecs = self.encoder.encode([e.text for e in entries])
            vecs = l2_normalize(np.asarray(vecs, dtype=np.float32))
            for entry, vec in zip(entries, vecs):
                entry.vec = vec
            n_clusters = None
            if len(entries) > self.brute_force_threshold:
                # 经验值：每簇约 sqrt(n) 个点；cluster_multiplier 用于往细里调（提高召回）
                n_clusters = max(2, int(np.sqrt(len(entries)) * self.cluster_multiplier))
            part = _Partition(level=lvl, entries=entries)
            part.finalize(self.encoder.dim, n_clusters, self.kmeans_iters, self.seed)
            self.partitions[lvl] = part
            if verbose:
                n_c = 0 if part.centroids is None else len(part.centroids)
                print(f"  level {lvl}: {len(entries)} 条，质心 {n_c or '（暴力扫描）'}")

        self.build_seconds = time.perf_counter() - t0
        return self

    # ---------------- 检索 ----------------
    def search(self, query: str, level: int, top_k: int = 5,
               n_probe: int | None = None,
               min_score: float | None = None) -> list[tuple[str, float, int]]:
        """级别 level 的检索。返回 [(object_id, score, level)]，按分数降序、按对象去重。

        min_score: 相似度下限，用于把"其实没有任何相关条款"识别出来。
                   不传则不做过滤（评测/校准场景需要看完整排序）。
        """
        q = l2_normalize(np.asarray(self.encoder.encode([query], is_query=True),
                                    dtype=np.float32))[0]
        n_probe = self.n_probe if n_probe is None else n_probe
        merged: dict[str, tuple[float, int]] = {}
        scanned = 0
        searched: list[int] = []
        for lvl in sorted(self.partitions):
            if lvl > level:
                continue                      # ← 结构性隔离：更高级别分区根本不可达
            searched.append(lvl)
            hits, n = self.partitions[lvl].search(q, n_probe, top_k, min_score=min_score)
            scanned += n
            for oid, score, hit_level in hits:
                prev = merged.get(oid)
                if prev is None or score > prev[0]:
                    merged[oid] = (score, hit_level)
        self.last_scanned = scanned
        self.last_searched_levels = searched
        ranked = sorted(merged.items(), key=lambda kv: -kv[1][0])[:top_k]
        return [(oid, score, hit_level) for oid, (score, hit_level) in ranked]

    def exact_search(self, query: str, level: int,
                     top_k: int = 5) -> list[tuple[str, float, int]]:
        """暴力精确检索（仅用于算 recall 基准，不用于线上）。"""
        q = l2_normalize(np.asarray(self.encoder.encode([query], is_query=True),
                                    dtype=np.float32))[0]
        merged: dict[str, tuple[float, int]] = {}
        for lvl in sorted(self.partitions):
            if lvl > level:
                continue
            part = self.partitions[lvl]
            scores = part.vecs @ q
            for i in np.argsort(-scores):
                oid = part.entries[i].object_id
                sc = float(scores[i])
                prev = merged.get(oid)
                if prev is None or sc > prev[0]:
                    merged[oid] = (sc, lvl)
        ranked = sorted(merged.items(), key=lambda kv: -kv[1][0])[:top_k]
        return [(oid, score, hit_level) for oid, (score, hit_level) in ranked]

    # ---------------- 校准 ----------------
    def calibrate_n_probe(self, queries: list[str], level: int, target_recall: float = 0.95,
                          top_k: int = 5, reference: dict[str, list[str]] | None = None,
                          ladder: tuple[int, ...] = (1, 2, 4, 8, 16, 32,
                                                     64, 128, 256, 512)) -> dict:
        """用暴力检索作基准，找出**达到目标召回所需的最小 n_probe**，并写入 self.n_probe。

        为什么必须校准而不是给个常数：n_probe 给小了会**静默**丢结果——
        实测 n_probe=2 在 1 万条规模上忠实度只有 0.31，即丢掉约 2/3 的命中，
        而系统看起来一切正常。这类"看起来在跑、其实在漏"的默认值最危险。

        建议构建后调用一次，把结果随索引一起持久化；换编码器或换语料后需重新校准。
        """
        if not queries:
            return {"n_probe": self.n_probe, "recall": None, "reached": False,
                    "reason": "no queries"}

        if reference is None:
            reference = {q: [oid for oid, _, _ in self.exact_search(q, level, top_k=top_k)]
                         for q in queries}

        part = self.partitions.get(level)
        limit = (len(part.centroids)
                 if part is not None and part.centroids is not None else 1)

        best: dict | None = None
        for n_probe in ladder:
            if n_probe > limit:
                break
            recalls = []
            for q in queries:
                got = [oid for oid, _, _ in self.search(q, level, top_k=top_k,
                                                       n_probe=n_probe)]
                truth = reference[q]
                recalls.append(len(set(got) & set(truth)) / max(1, len(truth)))
            recall = sum(recalls) / len(recalls)
            if best is None or recall > best["recall"]:
                best = {"n_probe": n_probe, "recall": round(recall, 4)}
            if recall >= target_recall:
                self.n_probe = n_probe
                return {"n_probe": n_probe, "recall": round(recall, 4),
                        "target": target_recall, "reached": True,
                        "max_probe": limit, "n_queries": len(queries)}
        if best:
            self.n_probe = best["n_probe"]
        return {**(best or {"n_probe": self.n_probe, "recall": 0.0}),
                "target": target_recall, "reached": False, "max_probe": limit,
                "n_queries": len(queries)}

    # ---------------- 统计 ----------------
    def allowed_entries(self, level: int) -> int:
        return sum(len(p.entries) for lvl, p in self.partitions.items() if lvl <= level)

    def stats(self) -> dict:
        return {
            "encoder": getattr(self.encoder, "name", type(self.encoder).__name__),
            "dim": int(self.encoder.dim),
            "build_seconds": round(self.build_seconds, 2),
            "n_entries": sum(len(p.entries) for p in self.partitions.values()),
            "per_level": {lvl: len(p.entries) for lvl, p in sorted(self.partitions.items())},
            "centroids_per_level": {
                lvl: (0 if p.centroids is None else len(p.centroids))
                for lvl, p in sorted(self.partitions.items())
            },
            "brute_force_threshold": self.brute_force_threshold,
            "cluster_multiplier": self.cluster_multiplier,
            "n_probe": self.n_probe,
        }
