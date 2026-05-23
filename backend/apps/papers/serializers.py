from rest_framework import serializers

from apps.papers.models import PaperRecord


class PaperRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaperRecord
        fields = [
            "id",
            "title",
            "keywords",
            "year",
            "summary",
            "corpus_id",
            "is_active",
            "source",
            "created_at",
            "updated_at",
        ]
