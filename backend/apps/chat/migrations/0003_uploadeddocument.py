from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0002_longtermmemory"),
    ]

    operations = [
        migrations.CreateModel(
            name="UploadedDocument",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("filename", models.CharField(max_length=255)),
                ("content_type", models.CharField(blank=True, default="", max_length=100)),
                ("extension", models.CharField(blank=True, default="", max_length=20)),
                ("extracted_text", models.TextField()),
                ("char_count", models.IntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "session",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="uploaded_documents",
                        to="chat.chatsession",
                    ),
                ),
            ],
            options={
                "db_table": "uploaded_document",
                "ordering": ["-created_at"],
            },
        ),
    ]
