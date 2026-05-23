from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.papers.models import PaperRecord as DBPaperRecord
from figure_agent.rag.paper_retriever import PaperRAGRetriever


class Command(BaseCommand):
    help = "Build/train FAISS vector index from active papers in a corpus."

    def add_arguments(self, parser):
        parser.add_argument(
            "--corpus-id",
            default="default",
            help="Corpus id to load from MySQL.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=500,
            help="MySQL pagination size.",
        )
        parser.add_argument(
            "--index-path",
            default="models_cache/faiss/paper_vectors.index",
            help="FAISS index output path.",
        )
        parser.add_argument(
            "--index-type",
            default="ivf_flat",
            choices=["ivf_flat", "flat"],
            help="FAISS index type.",
        )
        parser.add_argument(
            "--nlist",
            type=int,
            default=16,
            help="IVF cluster count target. Effective nlist may be reduced for small corpora.",
        )
        parser.add_argument(
            "--nprobe",
            type=int,
            default=4,
            help="Number of IVF clusters to probe during search.",
        )
        parser.add_argument(
            "--disable-bge",
            action="store_true",
            help="Use hash-ngram vectors instead of trying BGE-M3 embedding.",
        )

    def handle(self, *args, **options):
        corpus_id = options["corpus_id"].strip() or "default"
        page_size = max(1, options["page_size"])
        records = self._load_records(corpus_id=corpus_id, page_size=page_size)

        self.stdout.write(
            self.style.NOTICE(
                f"Building FAISS index: corpus_id={corpus_id}, records={len(records)}, "
                f"index_type={options['index_type']}, nlist={options['nlist']}, "
                f"nprobe={options['nprobe']}"
            )
        )

        index_path = Path(options["index_path"])
        if not index_path.is_absolute():
            index_path = settings.PROJECT_ROOT / index_path

        retriever = PaperRAGRetriever(
            enable_reranker=False,
            enable_bge_embedding=not options["disable_bge"],
            enable_faiss_vector_store=True,
            vector_index_path=str(index_path),
            vector_index_type=options["index_type"],
            vector_nlist=options["nlist"],
            vector_nprobe=options["nprobe"],
        )
        retriever.reload_from_records(records, force_rebuild_vector_store=True)

        store = retriever.vector_store
        if store is None or store.count() == 0:
            self.stdout.write(self.style.ERROR("FAISS index build failed: no vectors indexed."))
            return

        self.stdout.write(
            self.style.SUCCESS(
                "FAISS index built. "
                f"engine={store.engine}, index_type={store.index_type}, "
                f"trained={store.trained}, count={store.count()}, "
                f"dimension={store.dimension}, effective_nlist={store.effective_nlist}, "
                f"nprobe={store.nprobe}, path={store.index_path}"
            )
        )
        self.stdout.write(f"metadata={store.meta_path}")

    def _load_records(self, corpus_id: str, page_size: int):
        query = (
            DBPaperRecord.objects
            .filter(corpus_id=corpus_id, is_active=True)
            .exclude(title="")
            .exclude(summary="")
            .order_by("-id")
        )
        total = query.count()
        records = []
        offset = 0

        while offset < total:
            records.extend(list(query[offset: offset + page_size]))
            offset += page_size

        return records
