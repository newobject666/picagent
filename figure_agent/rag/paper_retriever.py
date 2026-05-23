# figure_agent/rag/paper_retriever.py

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from figure_agent.rag.bge_embedding import BGEEmbeddingModel
from figure_agent.rag.cross_reranker import BGEReranker
from figure_agent.rag.faiss_vector_store import FAISSVectorStore

import logging

logger = logging.getLogger("picagent")
@dataclass
class PaperRecord:
    """
    RAG 内部使用的论文记录。
    既支持从 papers.json 加载，也支持从 MySQL ORM 对象加载。
    """

    title: str
    keywords: List[str]
    year: str
    summary: str
    paper_id: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperRecord":
        return cls(
            paper_id=data.get("id") or data.get("paper_id"),
            title=str(data.get("title", "")).strip(),
            keywords=cls._normalize_keywords(data.get("keywords", [])),
            year=str(data.get("year", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
        )

    @classmethod
    def from_object(cls, obj: Any) -> "PaperRecord":
        """
        从 Django ORM 对象转换。
        例如 apps.papers.models.PaperRecord。
        """
        return cls(
            paper_id=getattr(obj, "id", None),
            title=str(getattr(obj, "title", "") or "").strip(),
            keywords=cls._normalize_keywords(getattr(obj, "keywords", [])),
            year=str(getattr(obj, "year", "") or "").strip(),
            summary=str(getattr(obj, "summary", "") or "").strip(),
        )

    @staticmethod
    def _normalize_keywords(raw_keywords: Any) -> List[str]:
        if raw_keywords is None:
            return []

        if isinstance(raw_keywords, list):
            return [
                str(keyword).strip()
                for keyword in raw_keywords
                if str(keyword).strip()
            ]

        if isinstance(raw_keywords, str):
            text = raw_keywords.strip()

            if not text:
                return []

            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return [
                        str(keyword).strip()
                        for keyword in data
                        if str(keyword).strip()
                    ]
            except Exception:
                pass

            return [
                item.strip()
                for item in text.replace("，", ",").split(",")
                if item.strip()
            ]

        return []


@dataclass
class PaperHit:
    """
    检索命中的论文。
    """

    paper: PaperRecord
    score: float
    matched_keywords: List[str]
    lexical_score: float = 0.0
    embedding_score: float = 0.0
    rerank_score: float = 0.0
    keyword_score: float = 0.0
    recall_sources: List[str] = field(default_factory=list)
    is_relevant: bool = False
    relevance_reason: str = ""


@dataclass
class BM25SearchResult:
    paper: PaperRecord
    score: float
    matched_keywords: List[str]


@dataclass
class VectorSearchResult:
    paper: PaperRecord
    score: float


@dataclass
class RagIntent:
    intent_type: str
    scope: str
    strategy: str
    required_evidence_count: int
    min_top_score: float
    min_score_gap: float
    allow_supplemental: bool


@dataclass
class RagEvidenceAssessment:
    status: str
    reason: str
    confidence: float
    needs_supplemental: bool = False
    requirements: List[str] = field(default_factory=list)
    coverage: Dict[str, bool] = field(default_factory=dict)
    missing_requirements: List[str] = field(default_factory=list)


@dataclass
class RagRetrievalResult:
    status: str
    intent: RagIntent
    hits: List[PaperHit]
    context_message: Optional[Dict[str, str]]
    assessment: RagEvidenceAssessment
    supplemental_queries: List[str] = field(default_factory=list)


class PaperRAGRetriever:
    """
    本地论文库 RAG 检索器。

    当前策略：
    1. 小论文库：直接全库 BGE rerank
    2. 大论文库：先宽召回候选池，再 BGE rerank
    3. 最终取 top3 注入 LLM 上下文

    这样 RAG 是“语义增强”，不是简单关键词匹配。
    """

    def __init__(
        self,
        paper_library_path: str = "PaperLibrary/papers.json",
        final_top_k: int = 3,
        candidate_top_k: int = 50,
        full_rerank_threshold: int = 200,
        max_summary_chars: int = 500,
        max_total_chars: int = 2500,
        enable_reranker: bool = True,
        reranker_use_fp16: bool = False,
        enable_bge_embedding: bool = True,
        embedding_model_path: Optional[str] = None,
        embedding_use_fp16: bool = False,
        enable_faiss_vector_store: bool = True,
        vector_index_path: Optional[str] = None,
        vector_index_type: str = "ivf_flat",
        vector_nlist: int = 32,
        vector_nprobe: int = 8,
        auto_reload_json: bool = False,
    ):
        self.paper_library_path = Path(paper_library_path)

        if not self.paper_library_path.is_absolute():
            self.paper_library_path = Path.cwd() / self.paper_library_path

        self.final_top_k = final_top_k
        self.candidate_top_k = candidate_top_k
        self.full_rerank_threshold = full_rerank_threshold
        self.max_summary_chars = max_summary_chars
        self.max_total_chars = max_total_chars
        self.supplemental_top_k = max(final_top_k * 2, 6)
        self.embedding_dim = 384
        self.embedding_weight = 0.38
        self.rerank_weight = 0.42
        self.keyword_weight = 0.20
        self.min_chunk_relevance = 0.34
        self.min_embedding_score = 0.18
        self.min_keyword_score = 0.18
        self.min_rerank_score = 0.30
        self.min_requirement_support = 0.32
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        self.rrf_k = 60

        self.papers: List[PaperRecord] = []
        self.embedding_model_name = "hash-ngram"
        self.embedding_model = None
        self.vector_store = None
        self.vector_store_model_name = "hash-ngram"

        if enable_bge_embedding:
            self.embedding_model = BGEEmbeddingModel(
                model_name_or_path=embedding_model_path,
                use_fp16=embedding_use_fp16,
            )
            self.embedding_model_name = self.embedding_model.model_name_or_path
            self.vector_store_model_name = self.embedding_model_name

        if enable_faiss_vector_store:
            index_path = Path(vector_index_path) if vector_index_path else Path("models_cache") / "faiss" / "paper_vectors.index"
            if not index_path.is_absolute():
                index_path = Path.cwd() / index_path

            self.vector_store = FAISSVectorStore(
                index_path=index_path,
                index_type=vector_index_type,
                nlist=vector_nlist,
                nprobe=vector_nprobe,
            )

        self.reranker = None
        if enable_reranker:
            self.reranker = BGEReranker(
                use_fp16=reranker_use_fp16,
                normalize=True,
                max_length=1024,
            )

        if auto_reload_json:
            self.reload()

    def reload_from_records(self, records, force_rebuild_vector_store: bool = False) -> None:
        """
        从 MySQL 查询结果或 dict 列表加载论文。

        records 可以是：
        1. Django ORM 对象列表
        2. dict 列表
        """
        self.papers = []

        if not records:
            logger.warning("RAG 从 records 加载论文库，但 records 为空")
            return

        for item in records:
            if isinstance(item, dict):
                paper = PaperRecord.from_dict(item)
            else:
                paper = PaperRecord.from_object(item)

            if paper.title and paper.summary:
                self.papers.append(paper)
            else:
                logger.warning(
                    f"跳过无效论文: title={bool(paper.title)}, "
                    f"summary={bool(paper.summary)}"
                )

        logger.info(f"RAG 已从 records 加载论文数量: {len(self.papers)}")
        self._rebuild_vector_store(force=force_rebuild_vector_store)

    def append_records_to_vector_store(self, records) -> int:
        """
        Incrementally append new paper vectors into the existing FAISS index.

        The existing IVF centroids are kept unchanged. FAISS assigns each new
        vector to the nearest trained centroid when ``add`` is called.
        """
        if self.vector_store is None or not self.vector_store.available:
            return 0

        self.vector_store.load_metadata()
        self._apply_vector_store_metadata()

        new_papers: List[PaperRecord] = []
        seen_keys = set(self.vector_store.paper_keys)

        for item in records:
            paper = PaperRecord.from_dict(item) if isinstance(item, dict) else PaperRecord.from_object(item)
            if not paper.title or not paper.summary:
                continue

            paper_key = self._paper_key(paper)
            if paper_key in seen_keys:
                continue

            seen_keys.add(paper_key)
            new_papers.append(paper)

        if not new_papers:
            return 0

        documents = [
            self._paper_to_rerank_text(paper)
            for paper in new_papers
        ]
        paper_keys = [
            self._paper_key(paper)
            for paper in new_papers
        ]
        existing_model = self.vector_store.metadata.get("embedding_model", "")
        allow_hash_fallback = not existing_model or str(existing_model).startswith("hash-ngram")
        vectors = self._encode_documents_for_vector_store(
            documents,
            allow_hash_fallback=allow_hash_fallback,
        )

        if not vectors:
            return 0

        added = self.vector_store.add(
            paper_keys=paper_keys,
            vectors=vectors,
            metadata={
                "embedding_model": self.vector_store_model_name,
                "source": "mysql_or_json_paper_records",
                "update_strategy": "incremental_add",
            },
        )
        self.papers.extend(new_papers[:added])
        return added

    def reload(self) -> None:
        """
        默认从 JSON 重新加载。
        命令行版本可用；Django Web 版推荐使用 reload_from_records。
        """
        self.papers = []

        logger.info(f"RAG 尝试从 JSON 加载论文库: {self.paper_library_path}")

        if not self.paper_library_path.exists():
            logger.warning(f"RAG JSON 文件不存在: {self.paper_library_path}")
            return

        with open(self.paper_library_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("papers.json 顶层必须是 list。")

        self.reload_from_records(data)


    def retrieve(
        self,
        query: str,
    ) -> List[PaperHit]:
        """
        完整 RAG 检索流程。

        小库：
            全部论文 → BGE rerank → top3

        大库：
            宽召回 top candidate_top_k → BGE rerank → top3
        """
        result = self.retrieve_with_guardrails(query)
        return result.hits[: self.final_top_k]

    def retrieve_with_guardrails(
        self,
        query: str,
    ) -> RagRetrievalResult:
        """
        RAG 防幻觉流程：
        意图识别 -> 检索范围/策略 -> 候选定位 -> 混合召回 -> 融合重排
        -> 证据充分性判断 -> 必要时补充检索 -> 带来源上下文/拒答。
        """
        query = query.strip()
        intent = self._recognize_intent(query)
        logger.info(
            "RAG guardrails query=%s intent=%s scope=%s strategy=%s papers=%s",
            query,
            intent.intent_type,
            intent.scope,
            intent.strategy,
            len(self.papers),
        )

        if not query or not self.papers:
            assessment = RagEvidenceAssessment(
                status="REFUSE",
                reason="论文库为空或查询为空，无法提供可靠证据。",
                confidence=0.0,
            )
            return RagRetrievalResult("REFUSE", intent, [], None, assessment)

        requirements = self._extract_requirements(query, intent)
        hits = self._hybrid_recall_and_rerank(query, intent)
        assessment = self.evidence_gate(query, hits, intent, requirements)
        supplemental_queries: List[str] = []

        if assessment.status == "RETRY" and intent.allow_supplemental:
            supplemental_queries = self._build_supplemental_queries(
                query=query,
                intent=intent,
                missing_requirements=assessment.missing_requirements,
            )
            supplemental_hits: List[PaperHit] = []

            for supplemental_query in supplemental_queries:
                supplemental_hits.extend(
                    self._hybrid_recall_and_rerank(supplemental_query, intent)
                )

            hits = self._merge_hits(hits + supplemental_hits)
            assessment = self.evidence_gate(query, hits, intent, requirements)

            if assessment.status == "RETRY":
                assessment = RagEvidenceAssessment(
                    status="REFUSE",
                    reason=(
                        "补充检索后证据仍不足，直接回答有较高幻觉风险。"
                        f"未覆盖需求: {', '.join(assessment.missing_requirements)}"
                    ),
                    confidence=assessment.confidence,
                    requirements=assessment.requirements,
                    coverage=assessment.coverage,
                    missing_requirements=assessment.missing_requirements,
                )

        final_hits = [
            hit
            for hit in hits
            if hit.is_relevant
        ][: self.final_top_k]
        context = self._format_guarded_context(final_hits, intent, assessment)
        context_message = {"role": "system", "content": context} if context else None

        return RagRetrievalResult(
            status=assessment.status,
            intent=intent,
            hits=final_hits,
            context_message=context_message,
            assessment=assessment,
            supplemental_queries=supplemental_queries,
        )

    def build_context_message(
        self,
        query: str,
    ) -> Optional[Dict[str, str]]:
        """
        构造注入 LLM 的 RAG 上下文。
        """
        return self.retrieve_with_guardrails(query).context_message

    def _recognize_intent(self, query: str) -> RagIntent:
        text = query.lower()

        if any(word in text for word in ("对比", "比较", "区别", "contrast", "compare")):
            return RagIntent(
                intent_type="comparison",
                scope="multi_paper",
                strategy="hybrid_broad",
                required_evidence_count=2,
                min_top_score=0.45,
                min_score_gap=0.03,
                allow_supplemental=True,
            )

        if any(word in text for word in ("综述", "调研", "survey", "overview", "总结")):
            return RagIntent(
                intent_type="survey",
                scope="topic",
                strategy="hybrid_broad",
                required_evidence_count=3,
                min_top_score=0.38,
                min_score_gap=0.0,
                allow_supplemental=True,
            )

        if any(word in text for word in ("精读", "细节", "指标", "实验结果", "消融", "ablation", "metric")):
            return RagIntent(
                intent_type="specific_fact",
                scope="focused",
                strategy="hybrid_precise",
                required_evidence_count=1,
                min_top_score=0.55,
                min_score_gap=0.04,
                allow_supplemental=True,
            )

        if any(word in text for word in ("创新", "创新点", "方法", "架构", "技术路线", "method", "architecture")):
            return RagIntent(
                intent_type="method_analysis",
                scope="focused",
                strategy="hybrid_precise",
                required_evidence_count=1,
                min_top_score=0.42,
                min_score_gap=0.02,
                allow_supplemental=True,
            )

        return RagIntent(
            intent_type="general",
            scope="topic",
            strategy="hybrid_balanced",
            required_evidence_count=1,
            min_top_score=0.36,
            min_score_gap=0.0,
            allow_supplemental=True,
        )

    def _hybrid_recall_and_rerank(
        self,
        query: str,
        intent: RagIntent,
    ) -> List[PaperHit]:
        candidates = self._build_hybrid_candidates(query, intent)

        if not candidates:
            return []

        logger.info(f"RAG hybrid candidates 数量: {len(candidates)}")

        if self.reranker is None:
            return self._normalize_and_sort_hits(candidates)

        try:
            reranked_hits = self._rerank_with_bge(query=query, candidates=candidates)
            return self._fuse_and_sort_hits(reranked_hits)
        except Exception as exc:
            logger.exception("BGE reranker 执行失败，降级为混合召回融合分数")
            print(f"▸ BGE reranker 执行失败，降级为混合召回：{exc}")
            return self._normalize_and_sort_hits(candidates)

    def _build_hybrid_candidates(
        self,
        query: str,
        intent: RagIntent,
    ) -> List[PaperHit]:
        recall_size = self.candidate_top_k
        if intent.strategy == "hybrid_broad":
            recall_size = max(self.candidate_top_k, self.final_top_k * 25)
        elif intent.strategy == "hybrid_precise":
            recall_size = max(self.final_top_k * 10, 20)

        if len(self.papers) <= self.full_rerank_threshold:
            recall_size = max(recall_size, len(self.papers))

        bm25_results = self._bm25_recall(query, top_k=recall_size)
        vector_results = self._vector_recall(query, top_k=recall_size)
        fused_items = self._rrf_fusion(bm25_results, vector_results, k=self.rrf_k)

        bm25_by_id = {
            self._paper_key(result.paper): result
            for result in bm25_results
        }
        vector_by_id = {
            self._paper_key(result.paper): result
            for result in vector_results
        }
        hits: List[PaperHit] = []

        for paper_key, rrf_score in fused_items[:recall_size]:
            paper = self._paper_by_key(paper_key)
            if paper is None:
                continue

            bm25_result = bm25_by_id.get(paper_key)
            vector_result = vector_by_id.get(paper_key)
            matched_keywords = (
                bm25_result.matched_keywords
                if bm25_result is not None
                else self._matched_keywords(query, paper)
            )
            bm25_score = bm25_result.score if bm25_result is not None else 0.0
            embedding_score = vector_result.score if vector_result is not None else self._embedding_similarity(
                query,
                self._paper_to_rerank_text(paper),
            )
            keyword_score = self._keyword_relevance_score(query, paper, matched_keywords)
            sources = []

            if bm25_result is not None:
                sources.append("bm25")

            if vector_result is not None:
                sources.append("vector")

            if matched_keywords:
                sources.append("keyword")

            if self._title_term_hits(query, paper):
                sources.append("title")

            hits.append(
                PaperHit(
                    paper=paper,
                    score=rrf_score,
                    matched_keywords=matched_keywords,
                    lexical_score=bm25_score,
                    embedding_score=embedding_score,
                    keyword_score=keyword_score,
                    recall_sources=list(dict.fromkeys(sources)),
                )
            )

        logger.info(
            "RAG hybrid recall: bm25=%s vector=%s rrf=%s",
            len(bm25_results),
            len(vector_results),
            len(hits),
        )

        return hits

    def _bm25_recall(
        self,
        query: str,
        top_k: int,
    ) -> List[BM25SearchResult]:
        query_terms = self._tokenize_for_bm25(query)
        if not query_terms:
            return []

        paper_terms = {
            self._paper_key(paper): self._tokenize_for_bm25(self._paper_to_rerank_text(paper))
            for paper in self.papers
        }
        doc_count = len(paper_terms)
        avg_doc_len = (
            sum(len(terms) for terms in paper_terms.values()) / doc_count
            if doc_count > 0
            else 0.0
        )
        document_frequency: Dict[str, int] = {}

        for terms in paper_terms.values():
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        results: List[BM25SearchResult] = []

        for paper in self.papers:
            key = self._paper_key(paper)
            terms = paper_terms.get(key, [])
            if not terms:
                continue

            term_frequency: Dict[str, int] = {}
            for term in terms:
                term_frequency[term] = term_frequency.get(term, 0) + 1

            score = 0.0
            doc_len = len(terms)

            for term in query_terms:
                tf = term_frequency.get(term, 0)
                if tf == 0:
                    continue

                df = document_frequency.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                numerator = tf * (self.bm25_k1 + 1)
                denominator = tf + self.bm25_k1 * (
                    1 - self.bm25_b + self.bm25_b * doc_len / max(avg_doc_len, 1.0)
                )
                score += idf * numerator / max(denominator, 1e-9)

            matched_keywords = self._matched_keywords(query, paper)
            if score > 0 or matched_keywords:
                results.append(
                    BM25SearchResult(
                        paper=paper,
                        score=score,
                        matched_keywords=matched_keywords,
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _vector_recall(
        self,
        query: str,
        top_k: int,
    ) -> List[VectorSearchResult]:
        vector_store_results = self._faiss_vector_recall(query=query, top_k=top_k)
        if vector_store_results:
            return vector_store_results

        documents = [
            self._paper_to_rerank_text(paper)
            for paper in self.papers
        ]

        if self.embedding_model is not None and not self.embedding_model.disabled:
            try:
                scores = self.embedding_model.score(query, documents)
                results = [
                    VectorSearchResult(
                        paper=paper,
                        score=float(score),
                    )
                    for paper, score in zip(self.papers, scores)
                    if float(score) > 0
                ]
                results.sort(key=lambda item: item.score, reverse=True)
                return results[:top_k]
            except Exception as exc:
                self.embedding_model_name = (
                    f"hash-ngram fallback ({self.embedding_model.model_name_or_path} unavailable)"
                )
                logger.warning(
                    "BGE-M3 embedding 召回失败，降级为 hash embedding: %s",
                    exc,
                )

        results: List[VectorSearchResult] = []

        for paper, document in zip(self.papers, documents):
            score = self._hash_embedding_similarity(query, document)

            if score > 0:
                results.append(
                    VectorSearchResult(
                        paper=paper,
                        score=score,
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _faiss_vector_recall(
        self,
        query: str,
        top_k: int,
    ) -> List[VectorSearchResult]:
        if self.vector_store is None or not self.vector_store.available:
            return []

        if self.vector_store.count() != len(self.papers):
            self._rebuild_vector_store()

        if self.vector_store.count() == 0:
            return []

        query_vector = self._encode_query_for_vector_store(query)
        if not query_vector:
            return []
        if self.vector_store.dimension and len(query_vector) != self.vector_store.dimension:
            logger.warning(
                "FAISS query 向量维度不匹配: query_dim=%s index_dim=%s",
                len(query_vector),
                self.vector_store.dimension,
            )
            return []

        results = []
        for paper_key, score in self.vector_store.search(query_vector, top_k=top_k):
            paper = self._paper_by_key(paper_key)
            if paper is None:
                continue

            results.append(
                VectorSearchResult(
                    paper=paper,
                    score=float(score),
                )
            )

        return results

    def _rrf_fusion(
        self,
        bm25_results: List[BM25SearchResult],
        vector_results: List[VectorSearchResult],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}

        for rank, item in enumerate(bm25_results, start=1):
            chunk_id = self._paper_key(item.paper)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)

        for rank, item in enumerate(vector_results, start=1):
            chunk_id = self._paper_key(item.paper)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _paper_key(self, paper: PaperRecord) -> str:
        return str(paper.paper_id or paper.title)

    def _paper_by_key(self, paper_key: str) -> Optional[PaperRecord]:
        for paper in self.papers:
            if self._paper_key(paper) == paper_key:
                return paper

        return None

    def _normalize_and_sort_hits(self, hits: List[PaperHit]) -> List[PaperHit]:
        max_lexical = max((hit.lexical_score for hit in hits), default=0.0)

        normalized_hits = []
        for hit in hits:
            lexical = hit.lexical_score / max_lexical if max_lexical > 0 else 0.0
            relevance_score, is_relevant, reason = self._evaluate_chunk_relevance(
                embedding_score=hit.embedding_score,
                rerank_score=hit.rerank_score,
                keyword_score=hit.keyword_score,
            )
            normalized_hits.append(
                PaperHit(
                    paper=hit.paper,
                    score=relevance_score,
                    matched_keywords=hit.matched_keywords,
                    lexical_score=lexical,
                    embedding_score=hit.embedding_score,
                    rerank_score=hit.rerank_score,
                    keyword_score=hit.keyword_score,
                    recall_sources=hit.recall_sources,
                    is_relevant=is_relevant,
                    relevance_reason=reason,
                )
            )

        normalized_hits.sort(key=lambda hit: hit.score, reverse=True)
        return normalized_hits

    def _fuse_and_sort_hits(self, hits: List[PaperHit]) -> List[PaperHit]:
        max_lexical = max((hit.lexical_score for hit in hits), default=0.0)

        fused_hits = []
        for hit in hits:
            lexical = hit.lexical_score / max_lexical if max_lexical > 0 else 0.0
            rerank = max(0.0, min(1.0, hit.rerank_score or hit.score))
            relevance_score, is_relevant, reason = self._evaluate_chunk_relevance(
                embedding_score=hit.embedding_score,
                rerank_score=rerank,
                keyword_score=hit.keyword_score,
            )
            sources = hit.recall_sources or []

            if "semantic_rerank" not in sources:
                sources = sources + ["semantic_rerank"]

            fused_hits.append(
                PaperHit(
                    paper=hit.paper,
                    score=relevance_score,
                    matched_keywords=hit.matched_keywords,
                    lexical_score=lexical,
                    embedding_score=hit.embedding_score,
                    rerank_score=rerank,
                    keyword_score=hit.keyword_score,
                    recall_sources=list(dict.fromkeys(sources)),
                    is_relevant=is_relevant,
                    relevance_reason=reason,
                )
            )

        fused_hits.sort(key=lambda hit: hit.score, reverse=True)
        return fused_hits

    def evidence_gate(
        self,
        query: str,
        hits: List[PaperHit],
        intent: RagIntent,
        requirements: List[str],
    ) -> RagEvidenceAssessment:
        """
        证据门控：PASS 可回答，RETRY 需要补充检索，REFUSE 明确拒答。
        """
        if not hits:
            return RagEvidenceAssessment(
                status="RETRY",
                reason="没有召回可用候选文档。",
                confidence=0.0,
                needs_supplemental=True,
                requirements=requirements,
                coverage={requirement: False for requirement in requirements},
                missing_requirements=requirements,
            )

        relevant_hits = [hit for hit in hits if hit.is_relevant]
        coverage = self._coverage_check(requirements, relevant_hits)
        missing_requirements = [
            requirement
            for requirement, covered in coverage.items()
            if not covered
        ]
        useful_hits = [
            hit
            for hit in relevant_hits
            if hit.is_relevant and hit.score >= intent.min_top_score
        ]
        top_score = hits[0].score
        second_score = hits[1].score if len(hits) > 1 else 0.0
        score_gap = top_score - second_score
        has_relevance_signal = any(
            hit.is_relevant
            and (
                hit.embedding_score >= self.min_embedding_score
                or hit.rerank_score >= self.min_rerank_score
                or hit.keyword_score >= self.min_keyword_score
            )
            for hit in hits[: self.final_top_k]
        )

        enough_count = len(useful_hits) >= intent.required_evidence_count
        enough_relevant_chunks = len(relevant_hits) >= 2
        enough_score = top_score >= intent.min_top_score
        enough_gap = score_gap >= intent.min_score_gap
        enough_coverage = not missing_requirements

        if (
            enough_score
            and enough_count
            and enough_relevant_chunks
            and has_relevance_signal
            and enough_coverage
        ):
            return RagEvidenceAssessment(
                status="PASS",
                reason="证据数量、相关性分数、来源信号和需求覆盖均满足当前意图要求。",
                confidence=min(1.0, top_score),
                requirements=requirements,
                coverage=coverage,
                missing_requirements=[],
            )

        if intent.allow_supplemental and missing_requirements:
            return RagEvidenceAssessment(
                status="RETRY",
                reason=(
                    "相关证据存在，但没有覆盖用户问题的全部需求点，触发补充检索。"
                    f"未覆盖需求: {', '.join(missing_requirements)}"
                ),
                confidence=min(1.0, top_score),
                needs_supplemental=True,
                requirements=requirements,
                coverage=coverage,
                missing_requirements=missing_requirements,
            )

        if intent.allow_supplemental and not enough_relevant_chunks:
            return RagEvidenceAssessment(
                status="RETRY",
                reason="相关证据片段少于 2 条，触发补充检索。",
                confidence=min(1.0, top_score),
                needs_supplemental=True,
                requirements=requirements,
                coverage=coverage,
                missing_requirements=missing_requirements or requirements,
            )

        if intent.allow_supplemental and (
            top_score >= intent.min_top_score * 0.75
            or has_relevance_signal
        ):
            return RagEvidenceAssessment(
                status="RETRY",
                reason="已有弱相关证据，但不足以直接回答，触发补充检索。",
                confidence=min(1.0, top_score),
                needs_supplemental=True,
                requirements=requirements,
                coverage=coverage,
                missing_requirements=missing_requirements,
            )

        return RagEvidenceAssessment(
            status="REFUSE",
            reason="召回证据不足，直接回答有较高幻觉风险。",
            confidence=min(1.0, top_score),
            requirements=requirements,
            coverage=coverage,
            missing_requirements=missing_requirements,
        )

    def _build_supplemental_queries(
        self,
        query: str,
        intent: RagIntent,
        missing_requirements: Optional[List[str]] = None,
    ) -> List[str]:
        terms = self._extract_terms(query)[:8]
        compact_terms = " ".join(terms)

        requirements = missing_requirements or self._extract_requirements(query, intent)
        queries = [
            *requirements,
            compact_terms or query,
        ]

        if intent.intent_type == "comparison":
            queries.append(f"{query} compare difference")
        elif intent.intent_type == "survey":
            queries.append(f"{query} survey overview")
        elif intent.intent_type == "specific_fact":
            queries.append(f"{query} result metric ablation")
        elif intent.intent_type == "method_analysis":
            queries.append(f"{query} method architecture innovation")
        else:
            queries.append(f"{query} evidence mechanism")

        seen = set()
        unique_queries = []

        for item in queries:
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                unique_queries.append(item)

        return unique_queries[:3]

    def _extract_requirements(
        self,
        query: str,
        intent: RagIntent,
    ) -> List[str]:
        """
        将用户问题拆成证据必须覆盖的小需求点。
        """
        normalized_query = query.strip()
        if not normalized_query:
            return []

        split_parts = re.split(
            r"[，,；;、/]|以及|并且|同时|还有|和|与|及|plus|and",
            normalized_query,
            flags=re.IGNORECASE,
        )
        requirements = [
            part.strip(" ：:。？！?!.")
            for part in split_parts
            if len(part.strip(" ：:。？！?!. ")) >= 2
        ]

        topic_terms = self._extract_terms(normalized_query)[:6]

        if intent.intent_type == "comparison":
            requirements.extend([
                f"{normalized_query} 的比较对象",
                f"{normalized_query} 的差异点",
            ])
        elif intent.intent_type == "survey":
            requirements.extend([
                f"{normalized_query} 的主要方向",
                f"{normalized_query} 的代表方法",
                f"{normalized_query} 的证据来源",
            ])
        elif intent.intent_type == "specific_fact":
            requirements.extend([
                f"{normalized_query} 的具体事实",
                f"{normalized_query} 的实验或指标证据",
            ])
        elif intent.intent_type == "method_analysis":
            requirements.extend([
                f"{normalized_query} 的方法",
                f"{normalized_query} 的创新点",
            ])

        if not requirements:
            requirements = [normalized_query]

        if len(requirements) == 1 and topic_terms:
            requirements.append(" ".join(topic_terms))

        seen = set()
        unique_requirements = []

        for requirement in requirements:
            requirement = re.sub(r"\s+", " ", requirement).strip()
            requirement_norm = self._normalize(requirement)

            if not requirement_norm or requirement_norm in seen:
                continue

            seen.add(requirement_norm)
            unique_requirements.append(requirement)

        return unique_requirements[:8]

    def _coverage_check(
        self,
        requirements: List[str],
        chunks: List[PaperHit],
    ) -> Dict[str, bool]:
        coverage: Dict[str, bool] = {}

        for requirement in requirements:
            coverage[requirement] = False

            for chunk in chunks:
                support_score = self._requirement_support_score(requirement, chunk)

                if support_score >= self.min_requirement_support:
                    coverage[requirement] = True
                    break

        return coverage

    def _requirement_support_score(
        self,
        requirement: str,
        hit: PaperHit,
    ) -> float:
        chunk_text = self._paper_to_rerank_text(hit.paper)
        embedding_score = self._embedding_similarity(requirement, chunk_text)
        keyword_score = self._keyword_relevance_score(
            requirement,
            hit.paper,
            self._matched_keywords(requirement, hit.paper),
        )
        rerank_score = 0.0

        if self.reranker is not None:
            try:
                scores = self.reranker.score(requirement, [chunk_text])
                rerank_score = scores[0] if scores else 0.0
            except Exception:
                rerank_score = hit.rerank_score
        else:
            rerank_score = hit.rerank_score

        support_score, _, _ = self._evaluate_chunk_relevance(
            embedding_score=embedding_score,
            rerank_score=rerank_score,
            keyword_score=keyword_score,
        )

        return support_score

    def _rebuild_vector_store(self, force: bool = False) -> None:
        """
        Rebuild FAISS vector index from the current in-memory paper records.

        MySQL stores the readable source records. FAISS stores a rebuildable
        vector index keyed by paper_id/title.
        """
        if self.vector_store is None or not self.vector_store.available:
            return

        if not self.papers:
            self.vector_store.clear()
            return

        documents = [
            self._paper_to_rerank_text(paper)
            for paper in self.papers
        ]
        paper_keys = [
            self._paper_key(paper)
            for paper in self.papers
        ]
        if not force and self._existing_vector_store_matches(paper_keys):
            self._apply_vector_store_metadata()
            logger.info(
                "FAISS 向量索引已存在且 paper_keys 匹配，跳过重建: count=%s model=%s",
                self.vector_store.count(),
                self.vector_store_model_name,
            )
            return

        vectors = self._encode_documents_for_vector_store(documents)

        if not vectors:
            logger.warning("FAISS 向量索引构建失败：没有可用 embedding")
            self.vector_store.clear()
            return

        self.vector_store.build(
            paper_keys=paper_keys,
            vectors=vectors,
            metadata={
                "embedding_model": self.vector_store_model_name,
                "source": "mysql_or_json_paper_records",
            },
        )
        logger.info(
            "FAISS 向量索引已构建: engine=%s count=%s model=%s",
            self.vector_store.engine,
            self.vector_store.count(),
            self.vector_store_model_name,
        )

    def _encode_documents_for_vector_store(
        self,
        documents: List[str],
        allow_hash_fallback: bool = True,
    ) -> List[List[float]]:
        prefer_hash = str(self.vector_store_model_name).startswith("hash-ngram")
        if (
            self.embedding_model is not None
            and not self.embedding_model.disabled
            and not prefer_hash
        ):
            try:
                vectors = self.embedding_model.encode(documents)
                if vectors:
                    self.vector_store_model_name = self.embedding_model.model_name_or_path
                    self.embedding_model_name = self.embedding_model.model_name_or_path
                    return vectors
            except Exception as exc:
                self.vector_store_model_name = (
                    f"hash-ngram fallback ({self.embedding_model.model_name_or_path} unavailable)"
                )
                self.embedding_model_name = self.vector_store_model_name
                logger.warning(
                    "BGE-M3 embedding 建库失败，FAISS 降级为 hash embedding: %s",
                    exc,
                )
                if not allow_hash_fallback:
                    return []

        self.vector_store_model_name = "hash-ngram"
        self.embedding_model_name = "hash-ngram"
        return [
            self._dense_hash_embedding(document)
            for document in documents
        ]

    def _existing_vector_store_matches(self, paper_keys: List[str]) -> bool:
        if self.vector_store is None or not self.vector_store.available:
            return False

        if not self.vector_store.load_metadata():
            return False

        if not self.vector_store.matches_keys(paper_keys):
            return False

        if self.vector_store.engine == "faiss" and not self.vector_store.index_path.exists():
            return False

        if self.vector_store.engine == "numpy_fallback" and not self.vector_store.numpy_path.exists():
            return False

        return True

    def _apply_vector_store_metadata(self) -> None:
        if self.vector_store is None:
            return

        embedding_model = self.vector_store.metadata.get("embedding_model")
        if embedding_model:
            self.vector_store_model_name = str(embedding_model)
            self.embedding_model_name = str(embedding_model)

    def _encode_query_for_vector_store(self, query: str) -> List[float]:
        if (
            self.embedding_model is not None
            and not self.embedding_model.disabled
            and not self.vector_store_model_name.startswith("hash-ngram")
        ):
            try:
                vectors = self.embedding_model.encode([query])
                if vectors:
                    return vectors[0]
            except Exception as exc:
                logger.warning("BGE-M3 query embedding 失败，跳过 FAISS BGE 查询: %s", exc)
                return []

        return self._dense_hash_embedding(query)

    def _merge_hits(self, hits: List[PaperHit]) -> List[PaperHit]:
        merged: Dict[str, PaperHit] = {}

        for hit in hits:
            key = str(hit.paper.paper_id or hit.paper.title)
            old_hit = merged.get(key)

            if old_hit is None or hit.score > old_hit.score:
                merged[key] = hit
            else:
                old_hit.recall_sources = list(
                    dict.fromkeys(old_hit.recall_sources + hit.recall_sources)
                )

        merged_hits = list(merged.values())
        merged_hits.sort(key=lambda hit: hit.score, reverse=True)
        return merged_hits

    def _evaluate_chunk_relevance(
        self,
        embedding_score: float,
        rerank_score: float,
        keyword_score: float,
    ) -> Tuple[float, bool, str]:
        """
        综合 Embedding 相似度、Reranker 分数、关键词命中判断 chunk 是否相关。
        """
        safe_embedding = max(0.0, min(1.0, embedding_score))
        safe_rerank = max(0.0, min(1.0, rerank_score))
        safe_keyword = max(0.0, min(1.0, keyword_score))

        if safe_rerank == 0.0:
            combined_score = safe_embedding * 0.60 + safe_keyword * 0.40
        else:
            combined_score = (
                safe_embedding * self.embedding_weight
                + safe_rerank * self.rerank_weight
                + safe_keyword * self.keyword_weight
            )

        strong_single_signal = (
            safe_embedding >= 0.42
            or safe_rerank >= 0.58
            or safe_keyword >= 0.65
        )
        two_signal_support = sum(
            [
                safe_embedding >= self.min_embedding_score,
                safe_rerank >= self.min_rerank_score,
                safe_keyword >= self.min_keyword_score,
            ]
        ) >= 2
        is_relevant = combined_score >= self.min_chunk_relevance and (
            strong_single_signal or two_signal_support
        )

        reason = (
            f"embedding={safe_embedding:.3f}, "
            f"reranker={safe_rerank:.3f}, "
            f"keyword={safe_keyword:.3f}, "
            f"combined={combined_score:.3f}"
        )

        if is_relevant:
            reason += "; passed relevance gate"
        else:
            reason += "; failed relevance gate"

        return combined_score, is_relevant, reason

    def _format_guarded_context(
        self,
        hits: List[PaperHit],
        intent: RagIntent,
        assessment: RagEvidenceAssessment,
    ) -> str:
        if assessment.status == "REFUSE":
            coverage_text = self._format_coverage(assessment)
            return (
                "本轮问题触发了本地论文库 RAG 防幻觉门控，但证据不足。\n"
                f"意图: {intent.intent_type}; 检索范围: {intent.scope}; 策略: {intent.strategy}\n"
                f"证据判断: {assessment.reason}\n"
                f"{coverage_text}\n"
                "请不要基于常识或猜测补全论文结论。应明确拒答或说明当前论文库证据不足，"
                "建议用户补充更具体的论文、关键词或上传相关文档。"
            )

        evidence = self._format_hits(hits)
        if not evidence:
            return ""

        return (
            "本轮问题触发了本地论文库 RAG 防幻觉策略。\n"
            f"意图: {intent.intent_type}; 检索范围: {intent.scope}; 策略: {intent.strategy}\n"
            f"证据状态: {assessment.status}; 置信度: {assessment.confidence:.2f}; 判断: {assessment.reason}\n"
            f"{self._format_coverage(assessment)}\n"
            f"{self._build_evidence_constrained_generation_rules()}\n"
            f"{evidence}"
        )

    @staticmethod
    def _build_evidence_constrained_generation_rules() -> str:
        return (
            "证据约束生成规则：\n"
            "1. 你只能基于下方【证据】片段回答，禁止使用模型参数知识、常识或外部资料自由补充。\n"
            "2. 证据没有明确提到的作者、方法细节、实验结果、指标、链接、结论，一律不能编写。\n"
            "3. 每个关键结论必须在句末标注来源，格式为 [来源: 证据N]；找不到对应证据就不能写成结论。\n"
            "4. 如果某个问题点没有证据支持，必须明确写“资料中未提到”，不要用猜测性表述补齐。\n"
            "5. 如果证据只支持部分回答，只回答已被证据支持的部分，并单独列出“资料中未提到”的部分。\n"
            "6. 如果用户要求具体数值、实验设置、对比结论，但证据片段没有给出，必须回答“资料中未提到”。\n"
            "7. 不要写“通常来说”“一般情况下”“可能是”等脱离证据的泛化内容。\n"
            "8. 推荐输出结构：先给“基于证据的回答”，再给“资料中未提到”。\n"
        )

    def _format_coverage(self, assessment: RagEvidenceAssessment) -> str:
        if not assessment.requirements:
            return "需求覆盖: 未拆解出额外需求点"

        lines = ["需求覆盖:"]

        for requirement in assessment.requirements:
            covered = assessment.coverage.get(requirement, False)
            mark = "已覆盖" if covered else "未覆盖"
            lines.append(f"- {requirement}: {mark}")

        return "\n".join(lines)

    def _build_candidates(
        self,
        query: str,
    ) -> List[PaperHit]:
        """
        构造候选论文池。

        如果论文库较小，直接全库进入 rerank。
        如果论文库较大，使用宽召回减少 reranker 开销。
        """
        if len(self.papers) <= self.full_rerank_threshold:
            return [
                PaperHit(
                    paper=paper,
                    score=0.0,
                    matched_keywords=self._matched_keywords(query, paper),
                )
                for paper in self.papers
            ]

        hits: List[PaperHit] = []

        for paper in self.papers:
            score, matched_keywords = self._wide_recall_score(
                query=query,
                paper=paper,
            )

            hits.append(
                PaperHit(
                    paper=paper,
                    score=score,
                    matched_keywords=matched_keywords,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)

        return hits[: self.candidate_top_k]

    def _rerank_with_bge(
        self,
        query: str,
        candidates: List[PaperHit],
    ) -> List[PaperHit]:
        """
        使用 BGE reranker 对候选论文做语义重排。
        """
        documents = [
            self._paper_to_rerank_text(hit.paper)
            for hit in candidates
        ]

        scores = self.reranker.score(
            query=query,
            documents=documents,
        )

        reranked_hits: List[PaperHit] = []

        for hit, score in zip(candidates, scores):
            reranked_hits.append(
                PaperHit(
                    paper=hit.paper,
                    score=float(score),
                    matched_keywords=hit.matched_keywords,
                    lexical_score=hit.lexical_score,
                    embedding_score=hit.embedding_score,
                    rerank_score=float(score),
                    keyword_score=hit.keyword_score,
                    recall_sources=hit.recall_sources,
                )
            )

        reranked_hits.sort(key=lambda hit: hit.score, reverse=True)

        return reranked_hits

    @staticmethod
    def _paper_to_rerank_text(paper: PaperRecord) -> str:
        """
        拼接给 BGE reranker 的候选论文文本。
        """
        return (
            f"Title: {paper.title}\n"
            f"Year: {paper.year}\n"
            f"Keywords: {', '.join(paper.keywords)}\n"
            f"Summary: {paper.summary}"
        )

    def _wide_recall_score(
        self,
        query: str,
        paper: PaperRecord,
    ) -> Tuple[float, List[str]]:
        """
        大库场景下的宽召回分数。

        注意：
        这个分数只用于减少候选池规模，不作为最终相关性判断。
        最终相关性由 BGE reranker 决定。
        """
        query_norm = self._normalize(query)
        title_norm = self._normalize(paper.title)
        summary_norm = self._normalize(paper.summary)
        keywords_norm = self._normalize(" ".join(paper.keywords))

        score = 0.0
        matched_keywords = self._matched_keywords(query, paper)

        # 关键词命中
        for keyword in matched_keywords:
            score += 10
            keyword_norm = self._normalize(keyword)

            if keyword_norm in title_norm:
                score += 3

        # 英文、数字、中文片段宽召回
        for term in self._extract_terms(query):
            term_norm = self._normalize(term)

            if len(term_norm) < 2:
                continue

            if term_norm in title_norm:
                score += 5

            if term_norm in keywords_norm:
                score += 4

            if term_norm in summary_norm:
                score += 1

        return score, matched_keywords

    def _matched_keywords(
        self,
        query: str,
        paper: PaperRecord,
    ) -> List[str]:
        """
        返回 query 命中的 paper keywords。
        """
        query_norm = self._normalize(query)

        matched = []

        for keyword in paper.keywords:
            keyword_norm = self._normalize(keyword)

            if keyword_norm and keyword_norm in query_norm:
                matched.append(keyword)

        return matched

    def _keyword_relevance_score(
        self,
        query: str,
        paper: PaperRecord,
        matched_keywords: List[str],
    ) -> float:
        query_terms = [
            self._normalize(term)
            for term in self._extract_terms(query)
            if len(self._normalize(term)) >= 2
        ]

        if not query_terms:
            return 0.0

        title_norm = self._normalize(paper.title)
        keyword_norm = self._normalize(" ".join(paper.keywords))
        summary_norm = self._normalize(paper.summary)
        matched_count = 0
        weighted_hits = 0.0

        for term in query_terms:
            if term in title_norm:
                matched_count += 1
                weighted_hits += 1.0
            elif term in keyword_norm:
                matched_count += 1
                weighted_hits += 0.85
            elif term in summary_norm:
                matched_count += 1
                weighted_hits += 0.35

        direct_keyword_bonus = min(0.35, len(matched_keywords) * 0.12)
        coverage = weighted_hits / max(1, len(query_terms))

        return max(0.0, min(1.0, coverage + direct_keyword_bonus))

    def _embedding_similarity(self, query: str, document: str) -> float:
        if self.embedding_model is not None and not self.embedding_model.disabled:
            try:
                scores = self.embedding_model.score(query, [document])
                if scores:
                    return scores[0]
            except Exception as exc:
                self.embedding_model_name = (
                    f"hash-ngram fallback ({self.embedding_model.model_name_or_path} unavailable)"
                )
                logger.warning(
                    "BGE-M3 embedding 相似度计算失败，降级为 hash embedding: %s",
                    exc,
                )

        return self._hash_embedding_similarity(query, document)

    def _hash_embedding_similarity(self, query: str, document: str) -> float:
        query_vector = self._hashed_embedding(query)
        document_vector = self._hashed_embedding(document)

        if not query_vector or not document_vector:
            return 0.0

        dot = 0.0
        for index, value in query_vector.items():
            dot += value * document_vector.get(index, 0.0)

        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        document_norm = math.sqrt(sum(value * value for value in document_vector.values()))

        if query_norm == 0.0 or document_norm == 0.0:
            return 0.0

        return max(0.0, min(1.0, dot / (query_norm * document_norm)))

    def _hashed_embedding(self, text: str) -> Dict[int, float]:
        terms = self._extract_terms(text)
        vector: Dict[int, float] = {}

        for term in terms:
            term_norm = self._normalize(term)
            if len(term_norm) < 2:
                continue

            digest = hashlib.md5(term_norm.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] = vector.get(index, 0.0) + sign

        return vector

    def _dense_hash_embedding(self, text: str) -> List[float]:
        sparse_vector = self._hashed_embedding(text)
        dense_vector = [0.0] * self.embedding_dim

        for index, value in sparse_vector.items():
            dense_vector[index] = float(value)

        norm = math.sqrt(sum(value * value for value in dense_vector))
        if norm == 0.0:
            return dense_vector

        return [
            value / norm
            for value in dense_vector
        ]

    def _format_hits(
        self,
        hits: List[PaperHit],
    ) -> str:
        """
        格式化 top3 论文信息。
        """
        chunks: List[str] = []
        total_chars = 0

        for index, hit in enumerate(hits, 1):
            paper = hit.paper

            summary = paper.summary
            if len(summary) > self.max_summary_chars:
                summary = summary[: self.max_summary_chars] + "..."

            chunk = (
                f"【证据{index}】\n"
                f"Paper ID：{paper.paper_id or 'N/A'}\n"
                f"Title：{paper.title}\n"
                f"Year：{paper.year or '未知'}\n"
                f"Keywords：{', '.join(paper.keywords) if paper.keywords else '未提供'}\n"
                f"Matched Keywords："
                f"{', '.join(hit.matched_keywords) if hit.matched_keywords else '无直接关键词命中'}\n"
                f"Recall Sources：{', '.join(hit.recall_sources) if hit.recall_sources else 'semantic'}\n"
                f"Evidence Score：{hit.score:.4f}\n"
                f"Embedding Model：{self.embedding_model_name}\n"
                f"Embedding Score：{hit.embedding_score:.4f}\n"
                f"Rerank Score：{hit.rerank_score:.4f}\n"
                f"Keyword Score：{hit.keyword_score:.4f}\n"
                f"Relevance Gate：{'PASS' if hit.is_relevant else 'FAIL'}; {hit.relevance_reason}\n"
                f"Summary：{summary}\n"
            )

            if total_chars + len(chunk) > self.max_total_chars:
                break

            chunks.append(chunk)
            total_chars += len(chunk)

        return "\n".join(chunks).strip()

    def _title_term_hits(
        self,
        query: str,
        paper: PaperRecord,
    ) -> int:
        title_norm = self._normalize(paper.title)
        count = 0

        for term in self._extract_terms(query):
            term_norm = self._normalize(term)
            if len(term_norm) >= 2 and term_norm in title_norm:
                count += 1

        return count

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        tokens: List[str] = []

        for term in self._extract_terms(text):
            term_norm = self._normalize(term)
            if len(term_norm) >= 2:
                tokens.append(term_norm)

        return tokens

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text.lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace("/", "")
            .strip()
        )

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """
        提取英文、数字和中文片段。

        目标不是精确分词，而是给大库宽召回使用。
        """
        terms: List[str] = []

        # 英文 / 数字片段
        terms.extend(re.findall(r"[a-zA-Z0-9]+", text))

        # 中文 2-4 gram
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_text = "".join(chinese_chars)

        for n in (2, 3, 4):
            for i in range(0, max(0, len(chinese_text) - n + 1)):
                terms.append(chinese_text[i:i + n])

        # 去重但保持顺序
        seen = set()
        unique_terms = []

        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def count(self) -> int:
        return len(self.papers)
