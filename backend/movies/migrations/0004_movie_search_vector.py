import django.contrib.postgres.indexes
import django.contrib.postgres.search
from django.db import migrations


def populate_search_vectors(apps, schema_editor):
    """Backfill the tsvector for rows imported before this column existed."""
    from movies.search_index import refresh_search_vectors

    refresh_search_vectors()


def noop(apps, schema_editor):
    """Nothing to undo: reversing the migration drops the column outright."""


class Migration(migrations.Migration):
    """Add the lexical retrieval channel: a weighted tsvector over
    title/director/actors, plus the GIN index that makes matching it cheap.
    """

    dependencies = [
        ("movies", "0003_remove_movie_embedding_hnsw"),
    ]

    operations = [
        migrations.AddField(
            model_name="movie",
            name="search_vector",
            field=django.contrib.postgres.search.SearchVectorField(
                blank=True, null=True
            ),
        ),
        migrations.AddIndex(
            model_name="movie",
            index=django.contrib.postgres.indexes.GinIndex(
                fields=["search_vector"], name="movie_search_vector_gin"
            ),
        ),
        migrations.RunPython(populate_search_vectors, noop),
    ]
