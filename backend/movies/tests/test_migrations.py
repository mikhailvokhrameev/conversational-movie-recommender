import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

APP = "movies"


def _leaf_migration(executor):
    (leaf,) = [node for node in executor.loader.graph.leaf_nodes() if node[0] == APP]
    return leaf


@pytest.mark.django_db(transaction=True)
class TestSessionTokenBackfillMigration:
    def test_backfill_assigns_distinct_tokens_to_preexisting_rows(self):
        executor = MigrationExecutor(connection)
        latest = _leaf_migration(executor)
        try:
            executor.migrate([(APP, "0001_initial")])
            executor.loader.build_graph()

            OldChatSession = executor.loader.project_state(
                (APP, "0001_initial")
            ).apps.get_model(APP, "ChatSession")
            OldChatSession.objects.create()
            OldChatSession.objects.create()
            OldChatSession.objects.create()

            executor = MigrationExecutor(connection)
            executor.migrate([(APP, "0002_chatsession_session_token")])
            executor.loader.build_graph()

            NewChatSession = executor.loader.project_state(
                (APP, "0002_chatsession_session_token")
            ).apps.get_model(APP, "ChatSession")
            tokens = list(NewChatSession.objects.values_list("session_token", flat=True))

            assert len(tokens) == 3
            assert all(tokens)
            assert len(set(tokens)) == 3
        finally:
            executor = MigrationExecutor(connection)
            executor.migrate([latest])
