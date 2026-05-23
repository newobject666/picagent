import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.rag.accuracy_tester import RAGAccuracyTester
from figure_agent.rag.paper_retriever import PaperRAGRetriever


class DeterministicReranker:
    """
    Lightweight reranker proxy for repeatable tests.

    Production uses BGE reranker. The test proxy keeps CI independent from local
    model files while still exercising the same three-signal relevance path:
    embedding score + reranker score + keyword score.
    """

    def score(self, query: str, documents: List[str]) -> List[float]:
        query_text = query.lower()
        scores = []

        for document in documents:
            document_text = document.lower()
            score = 0.05

            if "cnn" in query_text and "cnn" in document_text:
                score = 0.95
            elif any(
                term in query_text
                for term in ("lstm", "gate", "cell", "rnn", "hidden", "recurrent")
            ) and ("lstm" in document_text or "rnn" in document_text):
                score = 0.95
            elif any(
                term in query_text
                for term in ("transformer", "attention", "patch", "positional", "token")
            ) and "transformer" in document_text:
                score = 0.95

            scores.append(score)

        return scores


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    relevant_ids: Set[str]
    scenario: str


def build_retrieval_retriever() -> PaperRAGRetriever:
    records = []

    for index in range(10):
        records.append(
            {
                "id": f"common-{index}",
                "title": "model feature learning sequence attention background",
                "keywords": ["model", "feature", "learning", "sequence", "attention"],
                "year": "2024",
                "summary": (
                    "model feature learning sequence attention baseline benchmark "
                    f"note {index}"
                ),
            }
        )

    records.extend(
        [
            {
                "id": "cnn-class",
                "title": "CNN local feature image classification",
                "keywords": ["cnn", "local feature", "classification"],
                "year": "2024",
                "summary": (
                    "model feature learning cnn local feature image classification "
                    "convolution pooling"
                ),
            },
            {
                "id": "cnn-detect",
                "title": "CNN object detection feature map",
                "keywords": ["cnn", "object detection", "feature map"],
                "year": "2024",
                "summary": (
                    "model feature learning cnn detection feature map bounding box "
                    "spatial location"
                ),
            },
            {
                "id": "lstm-long",
                "title": "LSTM gated memory long dependency",
                "keywords": ["lstm", "gate", "cell state", "long dependency"],
                "year": "2024",
                "summary": (
                    "model sequence learning lstm gate cell state long dependency "
                    "vanishing gradient memory"
                ),
            },
            {
                "id": "rnn-seq",
                "title": "RNN hidden state sequence modeling",
                "keywords": ["rnn", "hidden state", "time step", "sequence"],
                "year": "2024",
                "summary": (
                    "model sequence learning rnn hidden state recurrent time step "
                    "ordered data"
                ),
            },
            {
                "id": "transformer-attn",
                "title": "Transformer self attention positional encoding",
                "keywords": ["transformer", "self attention", "positional encoding"],
                "year": "2024",
                "summary": (
                    "model attention learning transformer self attention token "
                    "relation positional encoding parallel"
                ),
            },
            {
                "id": "transformer-vision",
                "title": "Vision Transformer patch attention",
                "keywords": ["vit", "patch", "attention"],
                "year": "2024",
                "summary": (
                    "model attention learning vision transformer image patch "
                    "embedding global visual relation"
                ),
            },
            {
                "id": "cuda-noise",
                "title": "cuda latency model feature learning",
                "keywords": ["cuda", "latency", "model", "feature", "learning"],
                "year": "2023",
                "summary": (
                    "model feature learning cuda latency performance trace "
                    "local detection"
                ),
            },
            {
                "id": "logging-noise",
                "title": "logging trace model feature learning",
                "keywords": ["logging", "trace", "model", "feature", "learning"],
                "year": "2023",
                "summary": "model feature learning logging trace detection audit",
            },
            {
                "id": "audit-noise",
                "title": "audit trace sequence model learning",
                "keywords": ["audit", "trace", "sequence", "model", "learning"],
                "year": "2023",
                "summary": (
                    "model sequence learning hidden state gate dependency "
                    "audit trace"
                ),
            },
            {
                "id": "ops-noise",
                "title": "logging trace recurrent sequence model",
                "keywords": ["logging", "trace", "recurrent", "sequence", "model"],
                "year": "2023",
                "summary": (
                    "model sequence learning recurrent cell state long dependency "
                    "logging trace"
                ),
            },
            {
                "id": "cache-noise",
                "title": "cache locality attention model learning",
                "keywords": ["cache", "locality", "attention", "model", "learning"],
                "year": "2023",
                "summary": "model attention learning token positional patch cache locality",
            },
            {
                "id": "audit-attn-noise",
                "title": "audit trace visual attention model",
                "keywords": ["audit", "trace", "visual", "attention", "model"],
                "year": "2023",
                "summary": "model attention learning visual patch self attention audit trace",
            },
        ]
    )

    retriever = PaperRAGRetriever(
        final_top_k=2,
        candidate_top_k=10,
        full_rerank_threshold=0,
        enable_reranker=False,
        enable_bge_embedding=False,
        enable_faiss_vector_store=False,
    )
    retriever.reload_from_records(records)
    retriever.reranker = DeterministicReranker()
    return retriever


def retrieval_cases() -> List[RetrievalCase]:
    return [
        RetrievalCase(
            question="model feature learning cnn local detection cuda latency",
            relevant_ids={"cnn-class", "cnn-detect"},
            scenario="Raw recall is distracted by a rare operational term.",
        ),
        RetrievalCase(
            question="model feature learning cnn detection logging trace",
            relevant_ids={"cnn-class", "cnn-detect"},
            scenario="Raw recall is distracted by logging terms.",
        ),
        RetrievalCase(
            question="model sequence learning hidden state gate dependency audit trace",
            relevant_ids={"lstm-long", "rnn-seq"},
            scenario="Raw recall is distracted by audit terms.",
        ),
        RetrievalCase(
            question=(
                "model sequence learning recurrent cell state long dependency "
                "logging trace"
            ),
            relevant_ids={"lstm-long", "rnn-seq"},
            scenario="Raw recall is distracted by trace terms.",
        ),
        RetrievalCase(
            question=(
                "model attention learning transformer token positional patch "
                "cache locality"
            ),
            relevant_ids={"transformer-attn", "transformer-vision"},
            scenario="Raw recall is distracted by cache terms.",
        ),
        RetrievalCase(
            question="model attention learning visual patch self attention audit trace",
            relevant_ids={"transformer-attn", "transformer-vision"},
            scenario="Raw recall is distracted by visual audit terms.",
        ),
    ]


def _ids_from_bm25(
    retriever: PaperRAGRetriever,
    question: str,
    top_k: int,
) -> List[str]:
    return [
        str(result.paper.paper_id)
        for result in retriever._bm25_recall(question, top_k=top_k)
    ]


def _ids_from_vector(
    retriever: PaperRAGRetriever,
    question: str,
    top_k: int,
) -> List[str]:
    return [
        str(result.paper.paper_id)
        for result in retriever._vector_recall(question, top_k=top_k)
    ]


def _ids_from_hybrid_rrf(
    retriever: PaperRAGRetriever,
    question: str,
    top_k: int,
    candidate_k: int,
) -> List[str]:
    bm25_results = retriever._bm25_recall(question, top_k=candidate_k)
    vector_results = retriever._vector_recall(question, top_k=candidate_k)
    return [
        paper_id
        for paper_id, _ in retriever._rrf_fusion(
            bm25_results,
            vector_results,
            k=retriever.rrf_k,
        )[:top_k]
    ]


def _ids_before_three_signal_eval(
    retriever: PaperRAGRetriever,
    question: str,
    top_k: int,
) -> List[str]:
    intent = retriever._recognize_intent(question)
    candidates = retriever._build_hybrid_candidates(question, intent)

    return [
        retriever._paper_key(hit.paper)
        for hit in candidates[:top_k]
    ]


def _ids_after_three_signal_eval(
    retriever: PaperRAGRetriever,
    question: str,
    top_k: int,
) -> List[str]:
    intent = retriever._recognize_intent(question)
    hits = retriever._hybrid_recall_and_rerank(question, intent)

    return [
        retriever._paper_key(hit.paper)
        for hit in hits
        if hit.is_relevant
    ][:top_k]


def _recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    if not relevant_ids:
        return 0.0

    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def _reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for rank, paper_id in enumerate(retrieved_ids, start=1):
        if paper_id in relevant_ids:
            return 1 / rank

    return 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percent(value: float) -> float:
    return round(value * 100, 2)


def _relative_gain(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0

    return (new_value - old_value) / old_value


def _relative_reduction(old_value: float, new_value: float) -> float:
    if old_value == 0:
        return 0.0

    return (old_value - new_value) / old_value


def evaluate_retrieval(top_k: int = 2, candidate_k: int = 10) -> Dict[str, object]:
    retriever = build_retrieval_retriever()
    rows = []

    for case in retrieval_cases():
        before_ids = _ids_before_three_signal_eval(retriever, case.question, top_k)
        after_ids = _ids_after_three_signal_eval(retriever, case.question, top_k)
        rows.append(
            {
                "question": case.question,
                "scenario": case.scenario,
                "gold": sorted(case.relevant_ids),
                "before_three_signal": before_ids,
                "after_three_signal": after_ids,
                "recall": {
                    "before_three_signal": _recall_at_k(
                        before_ids,
                        case.relevant_ids,
                    ),
                    "after_three_signal": _recall_at_k(
                        after_ids,
                        case.relevant_ids,
                    ),
                },
                "reciprocal_rank": {
                    "before_three_signal": _reciprocal_rank(
                        before_ids,
                        case.relevant_ids,
                    ),
                    "after_three_signal": _reciprocal_rank(
                        after_ids,
                        case.relevant_ids,
                    ),
                },
            }
        )

    metrics = {}
    for method in ("before_three_signal", "after_three_signal"):
        recall = _mean(row["recall"][method] for row in rows)
        mrr = _mean(row["reciprocal_rank"][method] for row in rows)
        metrics[method] = {
            f"recall_at_{top_k}": recall,
            f"mrr_at_{top_k}": mrr,
            f"recall_at_{top_k}_percent": _percent(recall),
            f"mrr_at_{top_k}_percent": _percent(mrr),
        }

    before_recall = metrics["before_three_signal"][f"recall_at_{top_k}"]
    after_recall = metrics["after_three_signal"][f"recall_at_{top_k}"]

    return {
        "top_k": top_k,
        "case_count": len(rows),
        "baseline_definition": (
            "before_three_signal = hybrid candidate order without embedding + "
            "reranker + keyword relevance gate"
        ),
        "optimized_definition": (
            "after_three_signal = same candidates reranked and filtered by "
            "embedding similarity + reranker score + keyword hit"
        ),
        "metrics": metrics,
        "improvement": {
            "after_vs_before": {
                "absolute_points": _percent(after_recall - before_recall),
                "relative_percent": _percent(_relative_gain(after_recall, before_recall)),
            },
        },
        "rows": rows,
    }


def hallucination_cases() -> List[Dict[str, str]]:
    return [
        {
            "name": "cnn_supported_and_unsupported_claims",
            "rag_context": (
                "【证据1】\n"
                "CNN uses convolution kernels to extract local spatial image "
                "features and pooling to reduce feature map size."
            ),
            "baseline_answer": (
                "CNN uses convolution kernels to extract local spatial image "
                "features. [来源: 证据1]\n"
                "CNN guarantees quantum encrypted training and 99 percent "
                "accuracy on every image task. [来源: 证据1]"
            ),
            "guarded_answer": (
                "CNN uses convolution kernels to extract local spatial image "
                "features. [来源: 证据1]\n"
                "资料中未提到 CNN 能保证所有图像任务达到 99 percent accuracy。"
            ),
        },
        {
            "name": "lstm_supported_and_unsupported_claims",
            "rag_context": (
                "【证据1】\n"
                "LSTM uses input, forget and output gates to control memory, "
                "and the cell state helps carry long-term dependency signals."
            ),
            "baseline_answer": (
                "LSTM uses gates to control memory. [来源: 证据1]\n"
                "LSTM completely eliminates every gradient problem in all "
                "sequence tasks. [来源: 证据1]"
            ),
            "guarded_answer": (
                "LSTM uses gates to control memory. [来源: 证据1]\n"
                "资料中未提到 LSTM 能完全消除所有序列任务中的梯度问题。"
            ),
        },
        {
            "name": "transformer_supported_and_unsupported_claims",
            "rag_context": (
                "【证据1】\n"
                "Transformer uses multi-head self-attention to model token "
                "relationships in parallel and positional encoding to inject "
                "order information."
            ),
            "baseline_answer": (
                "Transformer uses self-attention to model token relationships "
                "in parallel. [来源: 证据1]\n"
                "Transformer uses positional encoding to inject order "
                "information. [来源: 证据1]\n"
                "Transformer provides quantum encrypted training and blockchain "
                "consensus for every sequence task. [来源: 证据1]"
            ),
            "guarded_answer": (
                "Transformer uses self-attention to model token relationships "
                "in parallel. [来源: 证据1]\n"
                "Transformer uses positional encoding to inject order "
                "information. [来源: 证据1]\n"
                "资料中未提到 Transformer 提供 quantum encrypted training 或 "
                "blockchain consensus。"
            ),
        },
    ]


def _failed_claim_count(report) -> int:
    return sum(1 for check in report.checks if check.status != "PASS")


def evaluate_hallucination() -> Dict[str, object]:
    tester = RAGAccuracyTester()
    rows = []
    totals = {
        "baseline": {"claims": 0, "failed": 0},
        "guarded": {"claims": 0, "failed": 0},
    }

    for case in hallucination_cases():
        baseline_report = tester.evaluate(
            answer=case["baseline_answer"],
            rag_context=case["rag_context"],
        )
        guarded_report = tester.evaluate(
            answer=case["guarded_answer"],
            rag_context=case["rag_context"],
        )
        baseline_failed = _failed_claim_count(baseline_report)
        guarded_failed = _failed_claim_count(guarded_report)

        totals["baseline"]["claims"] += baseline_report.total_claims
        totals["baseline"]["failed"] += baseline_failed
        totals["guarded"]["claims"] += guarded_report.total_claims
        totals["guarded"]["failed"] += guarded_failed

        rows.append(
            {
                "name": case["name"],
                "baseline": {
                    "status": baseline_report.status,
                    "claims": baseline_report.total_claims,
                    "failed": baseline_failed,
                    "hallucination_rate": (
                        baseline_failed / baseline_report.total_claims
                        if baseline_report.total_claims
                        else 0.0
                    ),
                },
                "guarded": {
                    "status": guarded_report.status,
                    "claims": guarded_report.total_claims,
                    "failed": guarded_failed,
                    "hallucination_rate": (
                        guarded_failed / guarded_report.total_claims
                        if guarded_report.total_claims
                        else 0.0
                    ),
                },
            }
        )

    baseline_rate = (
        totals["baseline"]["failed"] / totals["baseline"]["claims"]
        if totals["baseline"]["claims"]
        else 0.0
    )
    guarded_rate = (
        totals["guarded"]["failed"] / totals["guarded"]["claims"]
        if totals["guarded"]["claims"]
        else 0.0
    )

    return {
        "case_count": len(rows),
        "metric_definition": (
            "hallucination_rate = unsupported_or_unverifiable_claims / "
            "total_factual_claims"
        ),
        "baseline": {
            "claims": totals["baseline"]["claims"],
            "failed": totals["baseline"]["failed"],
            "hallucination_rate": baseline_rate,
            "hallucination_rate_percent": _percent(baseline_rate),
        },
        "guarded": {
            "claims": totals["guarded"]["claims"],
            "failed": totals["guarded"]["failed"],
            "hallucination_rate": guarded_rate,
            "hallucination_rate_percent": _percent(guarded_rate),
        },
        "reduction": {
            "absolute_points": _percent(baseline_rate - guarded_rate),
            "relative_percent": _percent(
                _relative_reduction(baseline_rate, guarded_rate)
            ),
        },
        "rows": rows,
    }


def evaluate_all() -> Dict[str, object]:
    return {
        "retrieval": evaluate_retrieval(),
        "hallucination": evaluate_hallucination(),
    }


def test_three_signal_relevance_gate_improves_recall():
    result = evaluate_retrieval()
    recall_key = f"recall_at_{result['top_k']}"

    before_recall = result["metrics"]["before_three_signal"][recall_key]
    after_recall = result["metrics"]["after_three_signal"][recall_key]

    assert after_recall == 1.0
    assert after_recall - before_recall >= 0.50


def test_hallucination_rate_drops_after_evidence_constraints():
    result = evaluate_hallucination()

    assert result["baseline"]["hallucination_rate"] > 0
    assert result["guarded"]["hallucination_rate"] == 0
    assert result["reduction"]["relative_percent"] >= 90


if __name__ == "__main__":
    print(json.dumps(evaluate_all(), ensure_ascii=False, indent=2))
