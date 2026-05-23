from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("papers", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="paperrecord",
            name="corpus_id",
            field=models.CharField(default="default", max_length=100, db_index=True),
        ),
        migrations.AddField(
            model_name="paperrecord",
            name="is_active",
            field=models.BooleanField(default=True, db_index=True),
        ),
        migrations.AddIndex(
            model_name="paperrecord",
            index=models.Index(fields=["corpus_id", "is_active"], name="paper_recor_corpus__a1b2c3_idx"),
        ),
        migrations.AddIndex(
            model_name="paperrecord",
            index=models.Index(fields=["corpus_id", "source"], name="paper_recor_corpus__d4e5f6_idx"),
        ),
    ]
