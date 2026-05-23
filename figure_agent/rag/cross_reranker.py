import os
from pathlib import Path
from typing import List, Optional


class BGEReranker:
    """
    BAAI/bge-reranker-v2-m3 重排器。

    作用：
    1. 接收用户问题 query
    2. 接收候选论文文本 documents
    3. 输出每篇候选论文的相关性分数

    注意：
    这里不是调用在线 API，而是本地加载 BGE reranker 模型。
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_fp16: bool = False,
        normalize: bool = True,
        max_length: int = 1024,
    ):
        self.model_name_or_path = model_name_or_path or self._resolve_model_path()
        self.use_fp16 = use_fp16
        self.normalize = normalize
        self.max_length = max_length
        self.model = None

    def _resolve_model_path(self) -> str:
        """
        解析模型路径。

        优先级：
        1. 环境变量 BGE_RERANKER_MODEL_PATH
        2. 项目本地 models/bge-reranker-v2-m3
        3. Hugging Face 模型名 BAAI/bge-reranker-v2-m3
        """

        env_path = os.environ.get("BGE_RERANKER_MODEL_PATH")
        if env_path:
            return env_path

        local_path = Path("models") / "bge-reranker-v2-m3"
        if local_path.exists():
            return str(local_path.resolve())

        return "BAAI/bge-reranker-v2-m3"

    def _load_model(self) -> None:
        """
        懒加载模型。
        """
        if self.model is not None:
            return

        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise ImportError(
                "未安装 FlagEmbedding，请先执行：pip install FlagEmbedding"
            ) from exc

        self.model = FlagReranker(
            self.model_name_or_path,
            use_fp16=self.use_fp16,
        )

    def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        对 query-document pairs 打分。
        """
        if not documents:
            return []

        self._load_model()

        pairs = [
            [query, document]
            for document in documents
        ]

        try:
            scores = self.model.compute_score(
                pairs,
                normalize=self.normalize,
                max_length=self.max_length,
            )
        except TypeError:
            # 兼容部分 FlagEmbedding 版本不支持 max_length 参数的情况
            scores = self.model.compute_score(
                pairs,
                normalize=self.normalize,
            )

        if isinstance(scores, float):
            return [scores]

        return [float(score) for score in scores]