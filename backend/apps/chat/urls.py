from django.urls import path

from apps.chat.views import (
    chat_view,
    session_list_view,
    session_messages_view,
    clear_session_view,
    memory_list_view,
    memory_delete_view,
    upload_document_view,
)

urlpatterns = [
    path("", chat_view),
    path("sessions/", session_list_view),
    path("sessions/<int:session_id>/messages/", session_messages_view),
    path("sessions/<int:session_id>/clear/", clear_session_view),
    path("memories/", memory_list_view),
    path("memories/<int:memory_id>/", memory_delete_view),
    path("documents/upload/", upload_document_view),
]
