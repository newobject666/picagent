import json
import logging
import hashlib
from pathlib import Path

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from apps.agents.agent_manage import agent_manager
from apps.papers.models import PaperRecord
from apps.papers.serializers import PaperRecordSerializer
from apps.papers.vector_index_update import PaperVectorIndexUpdater

logger = logging.getLogger("picagent")
DEFAULT_CORPUS_ID = "default"


def _bool_param(value, default=False) -> bool:
    if value is None:
        return default

    return str(value).lower() in {"1", "true", "yes", "on"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_paper_source(item, title: str, summary: str) -> str:
    explicit_source = (
        item.get("source")
        or item.get("arxiv_id")
        or item.get("id")
        or item.get("paper_id")
    )

    if explicit_source:
        source = str(explicit_source).strip()
        if source:
            return source[:100]

    digest = hashlib.md5(f"{title}\n{summary}".encode("utf-8")).hexdigest()[:16]
    return f"local_json:{digest}"


def sync_paper_records(
    data,
    corpus_id: str = DEFAULT_CORPUS_ID,
    clear: bool = False,
    deactivate_missing: bool = False,
):
    if not isinstance(data, list):
        return None, Response(
            {
                "detail": "papers.json 顶层必须是 list"
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    corpus_id = (corpus_id or DEFAULT_CORPUS_ID).strip() or DEFAULT_CORPUS_ID

    if clear:
        PaperRecord.objects.filter(corpus_id=corpus_id).delete()

    seen_sources = set()
    created = 0
    updated = 0
    unchanged = 0
    skipped = 0
    created_record_ids = []

    for item in data:
        if not isinstance(item, dict):
            skipped += 1
            continue

        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        keywords = item.get("keywords", [])
        year = str(item.get("year", "")).strip()

        if not title or not summary:
            skipped += 1
            continue

        source = build_paper_source(item, title, summary)
        seen_sources.add(source)
        record = PaperRecord.objects.filter(
            corpus_id=corpus_id,
            source=source,
        ).first()

        if record is None:
            record = PaperRecord.objects.create(
                title=title[:500],
                keywords=keywords,
                year=year,
                summary=summary,
                source=source,
                corpus_id=corpus_id,
                is_active=True,
            )
            created_record_ids.append(record.id)
            created += 1
        else:
            new_title = title[:500]
            changed = (
                record.title != new_title
                or record.keywords != keywords
                or record.year != year
                or record.summary != summary
                or not record.is_active
            )

            if changed:
                record.title = new_title
                record.keywords = keywords
                record.year = year
                record.summary = summary
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
    if deactivate_missing:
        deactivated = PaperRecord.objects.filter(
            corpus_id=corpus_id,
            is_active=True,
        ).exclude(source__in=seen_sources).update(is_active=False)

    vector_update = PaperVectorIndexUpdater().update_after_write(
        corpus_id=corpus_id,
        created_record_ids=created_record_ids,
        updated_count=updated,
        deactivated_count=deactivated,
        clear=clear,
    )
    agent_manager.reload_all_rag_from_mysql(corpus_id=corpus_id)

    return {
        "corpus_id": corpus_id,
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "skipped": skipped,
        "deactivated": deactivated,
        "vector_update": vector_update.to_dict(),
    }, None


@api_view(["GET"])
def paper_count_view(request):
    corpus_id = request.query_params.get("corpus_id", DEFAULT_CORPUS_ID)
    include_inactive = _bool_param(request.query_params.get("include_inactive"))
    papers = PaperRecord.objects.filter(corpus_id=corpus_id)

    if not include_inactive:
        papers = papers.filter(is_active=True)

    return Response(
        {
            "corpus_id": corpus_id,
            "include_inactive": include_inactive,
            "count": papers.count(),
        }
    )


@api_view(["GET"])
def paper_list_view(request):
    corpus_id = request.query_params.get("corpus_id", DEFAULT_CORPUS_ID)
    include_inactive = _bool_param(request.query_params.get("include_inactive"))
    page = max(1, int(request.query_params.get("page", 1)))
    page_size = min(500, max(1, int(request.query_params.get("page_size", 50))))
    start = (page - 1) * page_size
    end = start + page_size
    papers = PaperRecord.objects.filter(corpus_id=corpus_id)

    if not include_inactive:
        papers = papers.filter(is_active=True)

    papers = papers.order_by("-id")
    total = papers.count()
    serializer = PaperRecordSerializer(papers[start:end], many=True)
    return Response(
        {
            "corpus_id": corpus_id,
            "include_inactive": include_inactive,
            "page": page,
            "page_size": page_size,
            "total": total,
            "results": serializer.data,
        }
    )


@api_view(["POST"])
def sync_papers_from_json_view(request):
    """
    将 PaperLibrary/papers.json 同步到 MySQL。

    同步后：
    1. MySQL paper_record 更新
    2. 所有已存在 Agent 的 RAG 从 MySQL 重新加载
    """
    project_root = get_project_root()
    paper_path = project_root / "PaperLibrary" / "papers.json"

    logger.info(f"开始同步 papers.json: {paper_path}")

    if not paper_path.exists():
        return Response(
            {
                "detail": f"papers.json 不存在：{paper_path}"
            },
            status=status.HTTP_404_NOT_FOUND,
        )

    with open(paper_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    corpus_id = request.query_params.get("corpus_id", DEFAULT_CORPUS_ID)
    clear = _bool_param(request.query_params.get("clear"))
    deactivate_missing = _bool_param(request.query_params.get("deactivate_missing"))
    stats, error_response = sync_paper_records(
        data,
        corpus_id=corpus_id,
        clear=clear,
        deactivate_missing=deactivate_missing,
    )
    if error_response is not None:
        return error_response

    logger.info(f"papers.json 已同步到 MySQL, stats={stats}")

    return Response(
        {
            "detail": "同步完成，RAG 已从 MySQL 重新加载",
            **stats,
        }
    )


@api_view(["POST"])
def upload_papers_json_view(request):
    uploaded_file = request.FILES.get("file")

    if uploaded_file is None:
        return Response(
            {
                "detail": "请选择要上传的 papers.json 文件"
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not uploaded_file.name.lower().endswith(".json"):
        return Response(
            {
                "detail": "目前只支持上传 .json 文件"
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    raw_content = uploaded_file.read()

    try:
        data = json.loads(raw_content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return Response(
            {
                "detail": "文件不是有效的 UTF-8 JSON"
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    corpus_id = request.query_params.get("corpus_id", DEFAULT_CORPUS_ID)
    clear = _bool_param(request.query_params.get("clear"))
    deactivate_missing = _bool_param(request.query_params.get("deactivate_missing"))
    stats, error_response = sync_paper_records(
        data,
        corpus_id=corpus_id,
        clear=clear,
        deactivate_missing=deactivate_missing,
    )
    if error_response is not None:
        return error_response

    project_root = get_project_root()
    paper_library = project_root / "PaperLibrary"
    paper_library.mkdir(parents=True, exist_ok=True)
    paper_path = paper_library / "papers.json"
    paper_path.write_bytes(raw_content)

    logger.info(f"上传并同步 papers.json: {uploaded_file.name}, stats={stats}")

    return Response(
        {
            "detail": "上传完成，论文库和 RAG 已刷新",
            "filename": uploaded_file.name,
            **stats,
        }
    )


@api_view(["POST"])
def reload_rag_view(request):
    """
    从 MySQL 重新加载所有 Agent 的 RAG。
    """
    corpus_id = request.query_params.get("corpus_id", DEFAULT_CORPUS_ID)
    count = agent_manager.reload_all_rag_from_mysql(corpus_id=corpus_id)

    return Response(
        {
            "detail": "RAG 已从 MySQL 重新加载",
            "corpus_id": corpus_id,
            "count": count,
        }
    )
