import importlib

import pytest
from django.db import connection, migrations
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


migration_0008 = importlib.import_module("movies.migrations.0008_remove_chatsession_preference_vector")
OLD_UPDATED_AT_INDEX = "movies_chats_updated_5765c7_idx"
NEW_UPDATED_AT_INDEX = "movies_chat_updated_926282_idx"


def _index_operation():
    return next(
        op for op in migration_0008.Migration.operations
        if isinstance(op, migrations.SeparateDatabaseAndState)
    )


class TestUpdatedAtIndexMigration:
    """A database whose updated_at index is not named as 0001 says made the
    plain RenameIndex fail with 'relation ... does not exist'. The rename must
    work whether or not the old index is there."""

    def test_state_still_records_the_rename(self):
        (state_op,) = _index_operation().state_operations
        assert isinstance(state_op, migrations.RenameIndex)
        assert (state_op.old_name, state_op.new_name) == (OLD_UPDATED_AT_INDEX, NEW_UPDATED_AT_INDEX)

    def test_database_rename_tolerates_a_missing_old_index(self):
        (db_op,) = _index_operation().database_operations
        sql = " ".join(db_op.sql)
        assert f"ALTER INDEX IF EXISTS {OLD_UPDATED_AT_INDEX} RENAME TO {NEW_UPDATED_AT_INDEX}" in sql
        assert (
            f"CREATE INDEX IF NOT EXISTS {NEW_UPDATED_AT_INDEX} ON movies_chatsession (updated_at)" in sql
        )
