from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("movies", "0008_remove_chatsession_preference_vector"),
    ]

    operations = [
        # TMDB writes 0 where the runtime is unknown. The importer now stores
        # NULL for it; this fixes rows imported before. Not reversible: the
        # original zeros carry no information.
        migrations.RunSQL(
            sql="UPDATE movies_movie SET runtime = NULL WHERE runtime <= 0",
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
