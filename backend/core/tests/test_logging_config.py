import logging

import pytest


@pytest.mark.parametrize("name", ["core.ollama_client", "movies.views"])
def test_app_info_lines_reach_a_console_handler(name):
    """The per-call `Ollama call=...` and `turn_latency` lines are INFO. Without
    a LOGGING config only Python's last-resort handler exists, which drops
    everything below WARNING, so `docker compose logs backend` never showed them."""
    logger = logging.getLogger(name)
    assert logger.isEnabledFor(logging.INFO)
    handlers, node = [], logger
    while node:
        handlers += [h for h in node.handlers if isinstance(h, logging.StreamHandler)
                     and h.level <= logging.INFO and type(h).__module__ == "logging"]
        node = node.parent if node.propagate else None
    assert handlers, f"no console handler at INFO for {name}"
