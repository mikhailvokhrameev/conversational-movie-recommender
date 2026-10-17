from movies.management.commands.import_catalog import _row_to_movie


def _row(**overrides):
    row = {"id": "1", "title": "Test", "runtime": "95"}
    row.update(overrides)
    return row


class TestRuntimeMapping:
    def test_positive_runtime_is_kept(self):
        assert _row_to_movie(_row(runtime="95")).runtime == 95

    def test_zero_runtime_means_unknown(self):
        # TMDB writes 0 where the runtime is unknown.
        assert _row_to_movie(_row(runtime="0")).runtime is None

    def test_empty_runtime_means_unknown(self):
        assert _row_to_movie(_row(runtime="")).runtime is None
