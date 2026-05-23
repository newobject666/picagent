# figure_agent/rag/faiss_vector_store.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("picagent")


class FAISSVectorStore:
    """
    FAISS-backed vector index for paper chunks.

    MySQL remains the source of truth for text and metadata. This index stores
    dense vectors plus a paper-key mapping and can be rebuilt whenever the paper
    table changes.
    """

    def __init__(
        self,
        index_path: str | Path,
        index_type: str = "ivf_flat",
        nlist: int = 32,
        nprobe: int = 8,
        allow_numpy_fallback: bool = True,
    ):
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".meta.json")
        self.numpy_path = self.index_path.with_suffix(".npy")
        self.index_type = index_type
        self.nlist = max(1, nlist)
        self.nprobe = max(1, nprobe)
        self.allow_numpy_fallback = allow_numpy_fallback

        self.faiss = None
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.paper_keys: List[str] = []
        self.paper_clusters: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}
        self.dimension = 0
        self.engine = "unavailable"
        self.trained = False
        self.effective_nlist = 0
        self._metadata_loaded = False

        try:
            import faiss  # type: ignore

            self.faiss = faiss
            self.engine = "faiss"
        except Exception as exc:
            if allow_numpy_fallback:
                self.engine = "numpy_fallback"
                logger.warning("FAISS 不可用，向量库降级为 numpy 暴力检索: %s", exc)
            else:
                logger.warning("FAISS 不可用，向量索引关闭: %s", exc)

    @property
    def available(self) -> bool:
        return self.engine in {"faiss", "numpy_fallback"}

    def build(
        self,
        paper_keys: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.available or not paper_keys or not vectors:
            self.clear()
            return

        matrix = self._to_normalized_matrix(vectors)
        if matrix.size == 0:
            self.clear()
            return

        if len(paper_keys) != matrix.shape[0]:
            raise ValueError("paper_keys 数量必须和 vectors 数量一致")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.paper_keys = [str(key) for key in paper_keys]
        self.dimension = int(matrix.shape[1])
        self.vectors = matrix if self.engine == "numpy_fallback" else None
        self.metadata = metadata or {}

        if self.engine == "faiss":
            self.index = self._create_faiss_index(matrix)
            clusters = self._assign_clusters(matrix)
            self.index.add(matrix)
            self.faiss.write_index(self.index, str(self.index_path))
        else:
            clusters = [-1] * len(self.paper_keys)
            np.save(self.numpy_path, matrix)

        self.paper_clusters = {
            paper_key: cluster
            for paper_key, cluster in zip(self.paper_keys, clusters)
        }
        self._write_metadata(self.metadata)

    def add(
        self,
        paper_keys: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Append new vectors to an already trained index.

        For IVF indexes, FAISS assigns each added vector to the nearest trained
        centroid during ``index.add``. We also persist that centroid id in
        metadata so the update path can explain which cluster each new paper
        entered.
        """
        if not self.available or not paper_keys or not vectors:
            return 0

        self._ensure_metadata_loaded()
        matrix = self._to_normalized_matrix(vectors)
        if matrix.size == 0:
            return 0

        if len(paper_keys) != matrix.shape[0]:
            raise ValueError("paper_keys 数量必须和 vectors 数量一致")

        existing_keys = set(self.paper_keys)
        filtered_keys: List[str] = []
        filtered_vectors: List[np.ndarray] = []

        for key, vector in zip(paper_keys, matrix):
            key = str(key)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            filtered_keys.append(key)
            filtered_vectors.append(vector)

        if not filtered_keys:
            return 0

        matrix = np.asarray(filtered_vectors, dtype="float32")
        if self.dimension and matrix.shape[1] != self.dimension:
            raise ValueError(
                f"vector dimension mismatch: new={matrix.shape[1]} index={self.dimension}"
            )

        if not self.dimension:
            self.dimension = int(matrix.shape[1])

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine == "faiss":
            self._ensure_faiss_index_loaded()
            if self.index is None or not bool(getattr(self.index, "is_trained", True)):
                raise RuntimeError("FAISS index is not trained; rebuild before incremental add")

            clusters = self._assign_clusters(matrix)
            self.index.add(matrix)
            self.faiss.write_index(self.index, str(self.index_path))
        else:
            self._ensure_numpy_vectors_loaded()
            old_vectors = self.vectors
            self.vectors = matrix if old_vectors is None else np.vstack([old_vectors, matrix])
            clusters = [-1] * len(filtered_keys)
            np.save(self.numpy_path, self.vectors)

        self.paper_keys.extend(filtered_keys)
        for key, cluster in zip(filtered_keys, clusters):
            self.paper_clusters[key] = cluster

        if metadata:
            self.metadata.update(metadata)

        self._write_metadata(self.metadata)
        return len(filtered_keys)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        self._ensure_metadata_loaded()
        if not self.available or top_k <= 0 or not self.paper_keys:
            return []

        query_matrix = self._to_normalized_matrix([query_vector])
        if query_matrix.size == 0:
            return []

        if self.engine == "faiss":
            self._ensure_faiss_index_loaded()
            if self.index is None:
                return []

            if hasattr(self.index, "nprobe"):
                self.index.nprobe = min(self.nprobe, max(1, self.effective_nlist or self.nlist))

            scores, indices = self.index.search(query_matrix, min(top_k, len(self.paper_keys)))
            return self._format_results(scores[0], indices[0])

        self._ensure_numpy_vectors_loaded()
        if self.vectors is None:
            return []

        scores = self.vectors @ query_matrix[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.paper_keys[int(index)], float(scores[int(index)]))
            for index in top_indices
            if float(scores[int(index)]) > 0
        ]

    def clear(self) -> None:
        self.index = None
        self.vectors = None
        self.paper_keys = []
        self.paper_clusters = {}
        self.metadata = {}
        self.dimension = 0
        self.trained = False
        self.effective_nlist = 0
        self._metadata_loaded = True

    def _create_faiss_index(self, matrix: np.ndarray):
        if self.index_type == "flat" or matrix.shape[0] < 2:
            self.trained = True
            self.effective_nlist = 0
            return self.faiss.IndexFlatIP(self.dimension)

        if self.index_type != "ivf_flat":
            raise ValueError(f"Unsupported FAISS index_type: {self.index_type}")

        # Keep clusters reasonable for small corpora. For 1k papers this trains
        # around 32 clusters; for bigger corpora callers can raise nlist.
        effective_nlist = min(self.nlist, max(1, int(matrix.shape[0] ** 0.5)))
        quantizer = self.faiss.IndexFlatIP(self.dimension)
        index = self.faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            effective_nlist,
            self.faiss.METRIC_INNER_PRODUCT,
        )
        index.train(matrix)
        index.nprobe = min(self.nprobe, effective_nlist)
        self.trained = bool(index.is_trained)
        self.effective_nlist = effective_nlist
        return index

    def count(self) -> int:
        self._ensure_metadata_loaded()
        return len(self.paper_keys)

    def load_metadata(self) -> bool:
        if not self.meta_path.exists():
            self._metadata_loaded = True
            return False

        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.paper_keys = [str(key) for key in data.get("paper_keys", [])]
        self.paper_clusters = {
            str(key): int(value)
            for key, value in data.get("paper_clusters", {}).items()
        }
        self.metadata = dict(data.get("metadata", {}))
        self.dimension = int(data.get("dimension", self.dimension or 0))
        self.trained = bool(data.get("trained", self.trained))
        self.effective_nlist = int(data.get("effective_nlist", self.effective_nlist or 0))
        self.nlist = int(data.get("nlist", self.nlist))
        self.nprobe = int(data.get("nprobe", self.nprobe))
        self.index_type = str(data.get("index_type", self.index_type))
        self._metadata_loaded = True
        return True

    def matches_keys(self, paper_keys: Sequence[str]) -> bool:
        self._ensure_metadata_loaded()
        expected = {str(key) for key in paper_keys}
        return bool(expected) and expected == set(self.paper_keys)

    def _ensure_faiss_index_loaded(self) -> None:
        if self.index is not None:
            return

        self._ensure_metadata_loaded()
        if self.index_path.exists() and self.faiss is not None:
            self.index = self.faiss.read_index(str(self.index_path))
            self.effective_nlist = int(getattr(self.index, "nlist", self.effective_nlist or 0))
            self.trained = bool(getattr(self.index, "is_trained", True))
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = min(self.nprobe, max(1, self.effective_nlist or self.nlist))

    def _ensure_numpy_vectors_loaded(self) -> None:
        if self.vectors is not None:
            return

        self._ensure_metadata_loaded()
        if self.numpy_path.exists():
            self.vectors = np.load(self.numpy_path)

    def _ensure_metadata_loaded(self) -> None:
        if not self._metadata_loaded:
            self.load_metadata()

    def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        data = {
            "engine": self.engine,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "count": len(self.paper_keys),
            "trained": self.trained,
            "nlist": self.nlist,
            "effective_nlist": self.effective_nlist,
            "nprobe": self.nprobe,
            "paper_keys": self.paper_keys,
            "paper_clusters": self.paper_clusters,
            "metadata": metadata,
        }
        self.meta_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._metadata_loaded = True

    def _format_results(
        self,
        scores: Sequence[float],
        indices: Sequence[int],
    ) -> List[Tuple[str, float]]:
        results = []

        for score, index in zip(scores, indices):
            index = int(index)
            score = float(score)

            if index < 0 or index >= len(self.paper_keys) or score <= 0:
                continue

            results.append((self.paper_keys[index], score))

        return results

    def _assign_clusters(self, matrix: np.ndarray) -> List[int]:
        if self.engine != "faiss" or self.index is None:
            return [-1] * int(matrix.shape[0])

        quantizer = getattr(self.index, "quantizer", None)
        if quantizer is None:
            return [-1] * int(matrix.shape[0])

        _, indices = quantizer.search(matrix, 1)
        return [int(index) for index in indices[:, 0]]

    @staticmethod
    def _to_normalized_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype="float32")

        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return np.empty((0, 0), dtype="float32")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
