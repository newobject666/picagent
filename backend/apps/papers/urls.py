from django.urls import path

from apps.papers.views import (
    paper_count_view,
    paper_list_view,
    sync_papers_from_json_view,
    reload_rag_view,
    upload_papers_json_view,
)

urlpatterns = [
    path("count/", paper_count_view),
    path("", paper_list_view),
    path("sync/", sync_papers_from_json_view),
    path("reload-rag/", reload_rag_view),
    path("upload/", upload_papers_json_view),
]
