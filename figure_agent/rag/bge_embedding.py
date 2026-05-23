import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("picagent")


class BGEEmbeddingModel:
    """
    BAAI/bge-m3 embedding 模型封装。

    优先加载：
    1. 环境变量 BGE_EMBEDDING_MODEL_PATH
    2. 项目本地 models/bge-m3
    3. Hugging Face 模型名 BAAI/bge-m3

    如果依赖或模型不可用，调用方可以降级到本地 hash embedding。
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_fp16: bool = False,
        max_length: int = 8192,
        batch_size: int = 12,
    ):
        self.model_name_or_path = model_name_or_path or self._resolve_model_path()
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        self.disabled = False
        self.last_error = ""

    def _resolve_model_path(self) -> str:
        env_path = os.environ.get("BGE_EMBEDDING_MODEL_PATH")
        if env_path:
            return env_path

        local_path = Path("models") / "bge-m3"
        if local_path.exists():
            return str(local_path.resolve())

        return "BAAI/bge-m3"

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if self.disabled:
            raise RuntimeError(self.last_error or "BGE-M3 embedding model is disabled")

        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:
            self.disabled = True
            self.last_error = f"FlagEmbedding 或 torch 加载失败: {exc}"
            raise RuntimeError(self.last_error) from exc

        try:
            self.model = BGEM3FlagModel(
                self.model_name_or_path,
                use_fp16=self.use_fp16,
            )
            logger.info(f"BGE-M3 embedding 模型加载完成: {self.model_name_or_path}")
        except Exception as exc:
            self.disabled = True
            self.last_error = f"BGE-M3 embedding 模型加载失败: {exc}"
            raise RuntimeError(self.last_error) from exc

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        self._load_model()

        result = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        if isinstance(result, dict):
            vectors = result.get("dense_vecs")
        else:
            vectors = result

        if vectors is None:
            return []

        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        return [
            [float(value) for value in vector]
            for vector in vectors
        ]

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not query or not documents:
            return []

        vectors = self.encode([query, *documents])
        if len(vectors) < 2:
            return []

        query_vector = vectors[0]
        document_vectors = vectors[1:]

        return [
            self._cosine_similarity(query_vector, document_vector)
            for document_vector in document_vectors
        ]

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0

        for left_value, right_value in zip(left, right):
            dot += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value

        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0

        return max(0.0, min(1.0, dot / ((left_norm ** 0.5) * (right_norm ** 0.5))))
