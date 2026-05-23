import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from django.conf import settings

from apps.papers.models import PaperRecord as DBPaperRecord
from figure_agent.rag.paper_index_policy import choose_paper_index_update_mode
from figure_agent.rag.paper_retriever import PaperRAGRetriever

logger = logging.getLogger("picagent")


@dataclass
class PaperVectorIndexUpdateResult:
    corpus_id: str
    mode: str
    reason: str
    added_count: int
    updated_count: int
    deactivated_count: int
    indexed_count: int
    total_active_count: int
    deduped_count: int
    index_path: str
    index_type: str
    nlist: int
    nprobe: int
    embedding_model: str

    def to_dict(self):
        return asdict(self)


class PaperVectorIndexUpdater:
    """
    Keep the clustered FAISS index in sync with MySQL paper writes.

    Strategy:
    - Added papers <= 100: exact-dedupe by paper key, vectorize only the new
      records, and append them to the trained IVF index. FAISS assigns each new
      vector to its nearest centroid.
    - Added papers > 100, or any update/delete/clear: rebuild and retrain the
      IVF clustering index from the active corpus.
    """

    def __init__(
        self,
        index_path: Optional[str | Path] = None,
        index_type: str = "ivf_flat",
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
        incremental_limit: Optional[int] = None,
        page_size: Optional[int] = None,
        enable_bge_embedding: bool = True,
    ):
        self.index_path = self._resolve_index_path(
            index_path or os.environ.get("RAG_FAISS_INDEX_PATH") or "models_cache/faiss/paper_vectors.index"
        )
        self.index_type = index_type or os.environ.get("RAG_FAISS_INDEX_TYPE", "ivf_flat")
        self.nlist = nlist if nlist is not None else int(os.environ.get("RAG_FAISS_NLIST", "16"))
        self.nprobe = nprobe if nprobe is not None else int(os.environ.get("RAG_FAISS_NPROBE", "4"))
        self.incremental_limit = (
            incremental_limit
            if incremental_limit is not None
            else int(os.environ.get("RAG_FAISS_INCREMENTAL_LIMIT", "100"))
        )
        self.page_size = page_size if page_size is not None else int(os.environ.get("RAG_PAPER_PAGE_SIZE", "500"))
        self.enable_bge_embedding = enable_bge_embedding

    def update_after_write(
        self,
        corpus_id: str,
        created_record_ids: Iterable[int],
        updated_count: int = 0,
        deactivated_count: int = 0,
        clear: bool = False,
    ) -> PaperVectorIndexUpdateResult:
        corpus_id = (corpus_id or "default").strip() or "default"
        created_ids = list(dict.fromkeys(int(item) for item in created_record_ids if item))
        total_active_count = self._active_count(corpus_id)
        index_ready = self._existing_index_ready()
        decision = choose_paper_index_update_mode(
            added_count=len(created_ids),
            updated_count=updated_count,
            deactivated_count=deactivated_count,
            clear=clear,
            incremental_limit=self.incremental_limit,
            existing_index_ready=index_ready,
        )

        if decision.mode == "skip":
            return self._result(
                corpus_id=corpus_id,
                mode=decision.mode,
                reason=decision.reason,
                added_count=len(created_ids),
                updated_count=updated_count,
                deactivated_count=deactivated_count,
                indexed_count=0,
                total_active_count=total_active_count,
                deduped_count=0,
                embedding_model="unchanged",
            )

        if decision.mode == "incremental":
            try:
                return self._incremental_add(
                    corpus_id=corpus_id,
                    created_ids=created_ids,
                    decision_reason=decision.reason,
                    updated_count=updated_count,
                    deactivated_count=deactivated_count,
                    total_active_count=total_active_count,
                )
            except Exception as exc:
                logger.exception("FAISS incremental update failed; falling back to retrain")
                decision_reason = f"{decision.reason}; incremental failed, retrained instead: {exc}"
                return self._retrain(
                    corpus_id=corpus_id,
                    added_count=len(created_ids),
                    updated_count=updated_count,
                    deactivated_count=deactivated_count,
                    total_active_count=total_active_count,
                    reason=decision_reason,
                )

        return self._retrain(
            corpus_id=corpus_id,
            added_count=len(created_ids),
            updated_count=updated_count,
            deactivated_count=deactivated_count,
            total_active_count=total_active_count,
            reason=decision.reason,
        )

    def _incremental_add(
        self,
        corpus_id: str,
        created_ids: List[int],
        decision_reason: str,
        updated_count: int,
        deactivated_count: int,
        total_active_count: int,
    ) -> PaperVectorIndexUpdateResult:
        records = self._load_records(corpus_id=corpus_id, ids=created_ids)
        retriever = self._new_retriever()
        indexed_count = retriever.append_records_to_vector_store(records)
        deduped_count = max(0, len(records) - indexed_count)

        return self._result(
            corpus_id=corpus_id,
            mode="incremental",
            reason=decision_reason,
            added_count=len(created_ids),
            updated_count=updated_count,
            deactivated_count=deactivated_count,
            indexed_count=indexed_count,
            total_active_count=total_active_count,
            deduped_count=deduped_count,
            embedding_model=retriever.vector_store_model_name,
        )

    def _retrain(
        self,
        corpus_id: str,
        added_count: int,
        updated_count: int,
        deactivated_count: int,
        total_active_count: int,
        reason: str,
    ) -> PaperVectorIndexUpdateResult:
        records = self._load_records(corpus_id=corpus_id)
        retriever = self._new_retriever()
        retriever.reload_from_records(records, force_rebuild_vector_store=True)
        store = retriever.vector_store

        return self._result(
            corpus_id=corpus_id,
            mode="retrain",
            reason=reason,
            added_count=added_count,
            updated_count=updated_count,
            deactivated_count=deactivated_count,
            indexed_count=store.count() if store is not None else 0,
            total_active_count=total_active_count,
            deduped_count=0,
            embedding_model=retriever.vector_store_model_name,
        )

    def _new_retriever(self) -> PaperRAGRetriever:
        return PaperRAGRetriever(
            enable_reranker=False,
            enable_bge_embedding=self.enable_bge_embedding,
            enable_faiss_vector_store=True,
            vector_index_path=str(self.index_path),
            vector_index_type=self.index_type,
            vector_nlist=self.nlist,
            vector_nprobe=self.nprobe,
        )

    def _existing_index_ready(self) -> bool:
        retriever = self._new_retriever()
        store = retriever.vector_store
        if store is None or not store.available:
            return False

        if not store.load_metadata():
            return False

        if store.engine == "faiss" and not store.index_path.exists():
            return False

        if store.engine == "numpy_fallback" and not store.numpy_path.exists():
            return False

        return bool(store.trained and store.count() > 0)

    def _load_records(
        self,
        corpus_id: str,
        ids: Optional[List[int]] = None,
    ) -> List[DBPaperRecord]:
        query = (
            DBPaperRecord.objects
            .filter(corpus_id=corpus_id, is_active=True)
            .exclude(title="")
            .exclude(summary="")
        )

        if ids is not None:
            return list(query.filter(id__in=ids).order_by("-id"))

        query = query.order_by("-id")
        total = query.count()
        records: List[DBPaperRecord] = []
        offset = 0

        while offset < total:
            records.extend(list(query[offset: offset + self.page_size]))
            offset += self.page_size

        return records

    def _active_count(self, corpus_id: str) -> int:
        return (
            DBPaperRecord.objects
            .filter(corpus_id=corpus_id, is_active=True)
            .exclude(title="")
            .exclude(summary="")
            .count()
        )

    def _result(
        self,
        corpus_id: str,
        mode: str,
        reason: str,
        added_count: int,
        updated_count: int,
        deactivated_count: int,
        indexed_count: int,
        total_active_count: int,
        deduped_count: int,
        embedding_model: str,
    ) -> PaperVectorIndexUpdateResult:
        return PaperVectorIndexUpdateResult(
            corpus_id=corpus_id,
            mode=mode,
            reason=reason,
            added_count=added_count,
            updated_count=updated_count,
            deactivated_count=deactivated_count,
            indexed_count=indexed_count,
            total_active_count=total_active_count,
            deduped_count=deduped_count,
            index_path=str(self.index_path),
            index_type=self.index_type,
            nlist=self.nlist,
            nprobe=self.nprobe,
            embedding_model=embedding_model,
        )

    @staticmethod
    def _resolve_index_path(index_path: str | Path) -> Path:
        path = Path(index_path)
        if not path.is_absolute():
            path = settings.PROJECT_ROOT / path
        return path
