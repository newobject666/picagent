from django.urls import path, include

urlpatterns = [
    path("api/chat/", include("apps.chat.urls")),
    path("api/papers/", include("apps.papers.urls")),
]