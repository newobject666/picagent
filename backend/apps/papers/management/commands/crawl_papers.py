from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.papers.arxiv_crawler import fetch_arxiv_papers
from apps.papers.models import PaperRecord
from apps.papers.vector_index_update import PaperVectorIndexUpdater


class Command(BaseCommand):
    help = "Crawl paper metadata from arXiv and write records into MySQL paper_record."

    def add_arguments(self, parser):
        parser.add_argument(
            "--query",
            required=True,
            help="Search query, e.g. 'retrieval augmented generation'.",
        )
        parser.add_argument(
            "--raw-query",
            action="store_true",
            help=(
                "Use raw arXiv search syntax instead of wrapping query as all:<query>. "
                "Example: --raw-query --query 'cat:cs.CL AND all:RAG'"
            ),
        )
        parser.add_argument(
            "--max-results",
            type=int,
            default=20,
            help="Maximum number of papers to fetch.",
        )
        parser.add_argument(
            "--corpus-id",
            default="default",
            help="Target corpus id. Only this corpus is updated/reloaded.",
        )
        parser.add_argument(
            "--start",
            type=int,
            default=0,
            help="arXiv result offset.",
        )
        parser.add_argument(
            "--sort-by",
            default="submittedDate",
            choices=["relevance", "lastUpdatedDate", "submittedDate"],
            help="arXiv sort field.",
        )
        parser.add_argument(
            "--sort-order",
            default="descending",
            choices=["ascending", "descending"],
            help="arXiv sort order.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Fetch batch size. arXiv API is capped to 100 per request here.",
        )
        parser.add_argument(
            "--delay-seconds",
            type=float,
            default=3.0,
            help="Delay between paginated arXiv requests.",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=30,
            help="HTTP timeout in seconds.",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete existing paper_record rows in the target corpus before writing.",
        )
        parser.add_argument(
            "--deactivate-missing",
            action="store_true",
            help=(
                "Mark active records in the target corpus as inactive when their "
                "source is not returned by this crawl."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch and show papers without writing MySQL.",
        )
        parser.add_argument(
            "--reload-rag",
            action="store_true",
            help="Reload existing in-memory Agent RAG instances after writing MySQL.",
        )

    def handle(self, *args, **options):
        query = options["query"].strip()
        corpus_id = options["corpus_id"].strip() or "default"
        if not query:
            raise CommandError("--query cannot be empty")

        self.stdout.write(
            self.style.NOTICE(
                f"Fetching arXiv papers: query={query!r}, max={options['max_results']}, "
                f"corpus_id={corpus_id!r}"
            )
        )

        papers = fetch_arxiv_papers(
            query=query,
            max_results=options["max_results"],
            start=options["start"],
            sort_by=options["sort_by"],
            sort_order=options["sort_order"],
            batch_size=options["batch_size"],
            delay_seconds=options["delay_seconds"],
            timeout=options["timeout"],
            raw_query=options["raw_query"],
        )

        if not papers:
            self.stdout.write(self.style.WARNING("No papers fetched."))
            return

        if options["dry_run"]:
            for index, paper in enumerate(papers, start=1):
                self.stdout.write(
                    f"{index}. {paper.title} ({paper.year}) "
                    f"keywords={paper.keywords} source={paper.source}"
                )
            self.stdout.write(self.style.SUCCESS(f"Dry run fetched {len(papers)} papers."))
            return

        with transaction.atomic():
            if options["clear"]:
                deleted_count, _ = PaperRecord.objects.filter(corpus_id=corpus_id).delete()
                self.stdout.write(
                    self.style.WARNING(f"Cleared existing paper records: {deleted_count}")
                )

            created = 0
            updated = 0
            unchanged = 0
            seen_sources = set()
            created_record_ids = []

            for paper in papers:
                seen_sources.add(paper.source)
                record = PaperRecord.objects.filter(
                    corpus_id=corpus_id,
                    source=paper.source,
                ).first()

                if record is None:
                    record = PaperRecord.objects.create(
                        corpus_id=corpus_id,
                        is_active=True,
                        source=paper.source,
                        title=paper.title[:500],
                        keywords=paper.keywords,
                        year=paper.year,
                        summary=paper.summary,
                    )
                    created_record_ids.append(record.id)
                    created += 1
                else:
                    new_title = paper.title[:500]
                    changed = (
                        record.title != new_title
                        or record.keywords != paper.keywords
                        or record.year != paper.year
                        or record.summary != paper.summary
                        or not record.is_active
                    )

                    if changed:
                        record.title = new_title
                        record.keywords = paper.keywords
                        record.year = paper.year
                        record.summary = paper.summary
                        record.is_active = True
                        record.save(
                            update_fields=[
                                "title",
                                "keywords",
                                "year",
                                "summary",
                                "is_active",
                                "updated_at",
                            ]
                        )
                        updated += 1
                    else:
                        unchanged += 1

            deactivated = 0
            if options["deactivate_missing"]:
                deactivated = PaperRecord.objects.filter(
                    corpus_id=corpus_id,
                    is_active=True,
                ).exclude(source__in=seen_sources).update(is_active=False)

        vector_update = PaperVectorIndexUpdater().update_after_write(
            corpus_id=corpus_id,
            created_record_ids=created_record_ids,
            updated_count=updated,
            deactivated_count=deactivated,
            clear=options["clear"],
        )

        if options["reload_rag"]:
            from apps.agents.agent_manage import agent_manager

            agent_manager.reload_all_rag_from_mysql(corpus_id=corpus_id)

        self.stdout.write(
            self.style.SUCCESS(
                f"Crawl finished. corpus_id={corpus_id}, fetched={len(papers)}, "
                f"created={created}, updated={updated}, unchanged={unchanged}, "
                f"deactivated={deactivated}"
            )
        )
        self.stdout.write(f"Vector index update: {vector_update.to_dict()}")
        self.stdout.write(
            "MySQL stores paper text/metadata. FAISS vectors are incrementally added "
            "for small append batches and retrained for large or mutating batches."
        )
