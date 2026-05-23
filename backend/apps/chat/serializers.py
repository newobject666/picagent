from rest_framework import serializers

from apps.chat.models import ChatSession, ChatMessage, LongTermMemory, UploadedDocument


class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = [
            "id",
            "title",
            "current_task",
            "last_skill",
            "created_at",
            "updated_at",
        ]


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "session",
            "role",
            "content",
            "created_at",
        ]


class ChatRequestSerializer(serializers.Serializer):
    session_id = serializers.IntegerField(required=False, allow_null=True)
    message = serializers.CharField()
    document_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        allow_empty=True,
    )


class LongTermMemorySerializer(serializers.ModelSerializer):
    class Meta:
        model = LongTermMemory
        fields = [
            "id",
            "session",
            "category",
            "content",
            "importance",
            "is_active",
            "created_at",
            "updated_at",
        ]


class UploadedDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDocument
        fields = [
            "id",
            "session",
            "filename",
            "content_type",
            "extension",
            "char_count",
            "created_at",
        ]
