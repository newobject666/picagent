from django.db import models


class PaperRecord(models.Model):
    title = models.CharField(max_length=500)
    keywords = models.JSONField(default=list)
    year = models.CharField(max_length=20, blank=True, default="")
    summary = models.TextField()

    corpus_id = models.CharField(max_length=100, default="default", db_index=True)
    is_active = models.BooleanField(default=True, db_index=True)
    source = models.CharField(max_length=100, blank=True, default="local")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "paper_record"
        indexes = [
            models.Index(
                fields=["corpus_id", "is_active"],
                name="paper_recor_corpus__a1b2c3_idx",
            ),
            models.Index(
                fields=["corpus_id", "source"],
                name="paper_recor_corpus__d4e5f6_idx",
            ),
        ]

    def __str__(self):
        return self.title


class RagHitRecord(models.Model):
    query = models.TextField()
    paper = models.ForeignKey(
        PaperRecord,
        on_delete=models.CASCADE,
        related_name="rag_hits",
    )
    score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "rag_hit_record"
        ordering = ["-created_at"]
