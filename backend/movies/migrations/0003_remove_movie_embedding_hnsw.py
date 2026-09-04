from django.db import migrations


class Migration(migrations.Migration):
    """Drop the HNSW index on Movie.embedding in favour of exact cosine search.

    At ~18K rows an exact scan is fast enough to be irrelevant next to the
    LLM call in the same request, and it avoids both the approximate-recall
    loss and HNSW's degraded behaviour when hard filters are applied before
    the vector ordering.
    """

    dependencies = [
        ("movies", "0002_chatsession_session_token"),
    ]

    operations = [
        migrations.RemoveIndex(
            model_name="movie",
            name="movie_embedding_hnsw",
        ),
    ]
