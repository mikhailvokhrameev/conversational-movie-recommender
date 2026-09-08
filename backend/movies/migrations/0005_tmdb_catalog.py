import pgvector.django.vector
from django.db import migrations, models


class Migration(migrations.Migration):
    """Swap the catalog from Okko (Russian, 18K rows, parquet) to TMDB
    (English, ~50K rows, CSV) -- see docs/superpowers-adjacent plan notes.

    This is a full catalog replacement, not a merge: old Okko rows and new
    TMDB rows share no identity, so the tables are truncated first rather
    than migrated row-by-row. `ChatSession` is truncated too -- its
    `preference_vector` is the same width as `Movie.embedding` and can't be
    widened in place for existing 768-dim vectors, and stale session state
    tied to the old Russian catalog/genre-space has no value anyway (sessions
    already have a 24h TTL). This migration is not meaningfully reversible
    (the truncated data is gone), hence the no-op reverse.
    """

    dependencies = [
        ("movies", "0004_movie_search_vector"),
    ]

    operations = [
        migrations.RunSQL(
            "TRUNCATE TABLE movies_movie RESTART IDENTITY CASCADE; "
            "TRUNCATE TABLE movies_chatsession RESTART IDENTITY CASCADE;",
            reverse_sql=migrations.RunSQL.noop,
        ),
        migrations.RemoveField(model_name="movie", name="content_type"),
        migrations.RemoveField(model_name="movie", name="actors"),
        migrations.RemoveField(model_name="movie", name="director"),
        migrations.RemoveField(model_name="movie", name="age_rating"),
        migrations.RemoveField(model_name="movie", name="studio_name"),
        migrations.RemoveField(model_name="movie", name="url"),
        migrations.AddField(
            model_name="movie",
            name="tmdb_id",
            field=models.IntegerField(default=0, unique=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="movie",
            name="original_title",
            field=models.CharField(blank=True, default="", max_length=500),
        ),
        migrations.AddField(
            model_name="movie",
            name="original_language",
            field=models.CharField(blank=True, default="", max_length=10),
        ),
        migrations.AddField(
            model_name="movie",
            name="keywords",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="movie",
            name="runtime",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="movie",
            name="popularity",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="movie",
            name="vote_average",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="movie",
            name="vote_count",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="movie",
            name="poster_path",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
        migrations.AlterField(
            model_name="movie",
            name="embedding",
            field=pgvector.django.vector.VectorField(
                blank=True, dimensions=1024, null=True
            ),
        ),
        migrations.AlterField(
            model_name="chatsession",
            name="preference_vector",
            field=pgvector.django.vector.VectorField(
                blank=True, dimensions=1024, null=True
            ),
        ),
    ]
