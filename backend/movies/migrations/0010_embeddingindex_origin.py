from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("movies", "0009_null_zero_runtime")]

    operations = [
        migrations.AddField(
            model_name="embeddingindex",
            name="provider",
            field=models.CharField(default="local", max_length=32),
        ),
        migrations.AddField(
            model_name="embeddingindex",
            name="api_provider",
            field=models.CharField(blank=True, max_length=64, null=True),
        ),
    ]
