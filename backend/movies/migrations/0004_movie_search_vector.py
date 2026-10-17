import django.contrib.postgres.indexes
import django.contrib.postgres.search
from django.db import migrations


class Migration(migrations.Migration):
    """Add the lexical retrieval channel: a weighted tsvector over
    title/director/actors, plus the GIN index that makes matching it cheap.

    No backfill here: the very next migration (0005_tmdb_catalog) truncates
    movies_movie outright, and refresh_search_vectors() references columns
    (original_title, keywords) that only exist after that migration. Actual
    population happens via import_catalog, which calls
    refresh_search_vectors() once TMDB rows are loaded.
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
    ]
