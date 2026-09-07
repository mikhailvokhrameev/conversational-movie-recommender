import movies.models
from django.db import migrations, models


def backfill_session_token(apps, schema_editor):
    ChatSession = apps.get_model("movies", "ChatSession")
    for session in ChatSession.objects.filter(session_token__isnull=True).iterator():
        session.session_token = movies.models.generate_token()
        session.save(update_fields=["session_token"])


class Migration(migrations.Migration):

    dependencies = [
        ("movies", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="session_token",
            field=models.CharField(
                db_index=True,
                null=True,
                default=None,
                max_length=64,
            ),
        ),
        migrations.RunPython(backfill_session_token, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="chatsession",
            name="session_token",
            field=models.CharField(
                db_index=True,
                default=movies.models.generate_token,
                max_length=64,
            ),
        ),
    ]
