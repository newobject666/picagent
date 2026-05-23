import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from figure_agent.rag.faiss_vector_store import FAISSVectorStore
from figure_agent.rag.paper_retriever import PaperRAGRetriever


def test_paper_rag_builds_vector_store_from_text_records(tmp_path):
    records = [
        {
            "id": "cnn-detect",
            "title": "CNN object detection feature map",
            "keywords": ["cnn", "object detection", "feature map"],
            "year": "2024",
            "summary": "CNN feature maps support object detection and bounding box regression.",
        },
        {
            "id": "lstm-long",
            "title": "LSTM long dependency",
            "keywords": ["lstm", "gate", "cell state"],
            "year": "2024",
            "summary": "LSTM gates and cell state help model long term sequence dependency.",
        },
    ]
    retriever = PaperRAGRetriever(
        enable_reranker=False,
        enable_bge_embedding=False,
        enable_faiss_vector_store=True,
        vector_index_path=str(tmp_path / "paper_vectors.index"),
        full_rerank_threshold=0,
    )

    retriever.reload_from_records(records)
    results = retriever._vector_recall("object detection bounding box feature map", top_k=1)

    assert retriever.vector_store is not None
    assert retriever.vector_store.count() == 2
    assert retriever.vector_store.engine in {"faiss", "numpy_fallback"}
    assert results
    assert results[0].paper.paper_id == "cnn-detect"
    assert (tmp_path / "paper_vectors.meta.json").exists()


def test_mysql_records_remain_text_source_while_vectors_are_rebuildable(tmp_path):
    record = {
        "id": 1,
        "title": "Transformer attention",
        "keywords": ["transformer", "attention"],
        "year": "2024",
        "summary": "Transformer uses self-attention for token relationship modeling.",
    }
    retriever = PaperRAGRetriever(
        enable_reranker=False,
        enable_bge_embedding=False,
        enable_faiss_vector_store=True,
        vector_index_path=str(tmp_path / "paper_vectors.index"),
    )

    retriever.reload_from_records([record])

    paper = retriever.papers[0]
    assert paper.title == record["title"]
    assert paper.summary == record["summary"]
    assert paper.keywords == record["keywords"]
    assert retriever.vector_store.count() == 1


def test_faiss_ivf_incremental_add_assigns_new_paper_to_cluster(tmp_path):
    store = FAISSVectorStore(
        index_path=tmp_path / "paper_vectors.index",
        index_type="ivf_flat",
        nlist=2,
        nprobe=1,
        allow_numpy_fallback=False,
    )

    if store.engine != "faiss":
        pytest.skip("faiss is not installed")

    keys = [f"x-{index}" for index in range(40)] + [f"y-{index}" for index in range(40)]
    vectors = [[1.0, index / 1000.0] for index in range(40)]
    vectors += [[index / 1000.0, 1.0] for index in range(40)]

    store.build(keys, vectors, metadata={"embedding_model": "hash-ngram"})
    added = store.add(["new-x"], [[1.0, 0.2]], metadata={"update_strategy": "incremental_add"})
    results = store.search([1.0, 0.2], top_k=1)

    assert added == 1
    assert store.count() == 81
    assert store.paper_clusters["new-x"] >= 0
    assert results[0][0] == "new-x"


def test_faiss_incremental_add_dedupes_existing_paper_key(tmp_path):
    store = FAISSVectorStore(
        index_path=tmp_path / "paper_vectors.index",
        index_type="flat",
        allow_numpy_fallback=True,
    )
    store.build(["paper-1"], [[1.0, 0.0]], metadata={"embedding_model": "hash-ngram"})

    added = store.add(["paper-1", "paper-2"], [[1.0, 0.0], [0.0, 1.0]])

    assert added == 1
    assert store.count() == 2
    assert store.paper_keys.count("paper-1") == 1


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        test_paper_rag_builds_vector_store_from_text_records(Path(temp_dir) / "case1")
        test_mysql_records_remain_text_source_while_vectors_are_rebuildable(
            Path(temp_dir) / "case2"
        )

    print("FAISS vector store tests passed.")
