"""Pure decision logic for the embedding-index provenance record: no DB, no model."""

import pytest

from core.embedding_provenance import (
    IndexCheck,
    ProvenanceError,
    check_verdict,
    compare_provenance,
    current_provenance,
    plan_generate,
)

CURRENT = {
    "model_name": "BAAI/bge-m3",
    "dimensions": 1024,
    "text_version": 1,
    "text_config": {"description_chars": 500, "max_keywords": 10},
}


def _stored(**changes):
    stored = {**CURRENT, "text_config": dict(CURRENT["text_config"])}
    stored.update(changes)
    return stored


class TestCurrentProvenance:
    def test_reflects_settings_and_template_version(self):
        from django.conf import settings

        from core.embedding_service import EMBEDDING_TEXT_VERSION

        current = current_provenance()

        assert current == {
            "model_name": settings.EMBEDDING_MODEL,
            "dimensions": settings.EMBEDDING_DIMENSIONS,
            "text_version": EMBEDDING_TEXT_VERSION,
            "text_config": {
                "description_chars": settings.EMBEDDING_TEXT_DESCRIPTION_CHARS,
                "max_keywords": settings.EMBEDDING_TEXT_MAX_KEYWORDS,
            },
        }


class TestCompareProvenance:
    def test_identical_record_is_ok(self):
        check = compare_provenance(_stored(), CURRENT)

        assert check == IndexCheck("ok", [])

    def test_no_record_is_missing(self):
        check = compare_provenance(None, CURRENT)

        assert check.status == "missing"
        assert check.differences == []

    @pytest.mark.parametrize(
        "changes, field",
        [
            ({"model_name": "other/model"}, "model_name"),
            ({"dimensions": 768}, "dimensions"),
            ({"text_version": 2}, "text_version"),
            (
                {"text_config": {"description_chars": 400, "max_keywords": 10}},
                "text_config.description_chars",
            ),
            (
                {"text_config": {"description_chars": 500, "max_keywords": 5}},
                "text_config.max_keywords",
            ),
        ],
    )
    def test_each_field_difference_is_a_mismatch_naming_the_field(self, changes, field):
        check = compare_provenance(_stored(**changes), CURRENT)

        assert check.status == "mismatch"
        assert len(check.differences) == 1
        assert field in check.differences[0]

    def test_difference_states_both_the_stored_and_the_current_value(self):
        check = compare_provenance(_stored(model_name="old/model"), CURRENT)

        assert "old/model" in check.differences[0]
        assert "BAAI/bge-m3" in check.differences[0]

    def test_all_differing_fields_are_listed(self):
        check = compare_provenance(_stored(model_name="x", dimensions=1, text_version=9), CURRENT)

        assert check.status == "mismatch"
        assert len(check.differences) == 3

    def test_text_config_key_missing_from_the_stored_record_is_a_difference(self):
        check = compare_provenance(_stored(text_config={"description_chars": 500}), CURRENT)

        assert check.status == "mismatch"
        assert "text_config.max_keywords" in check.differences[0]

    def test_stored_extra_text_config_key_is_a_difference(self):
        stored = _stored(text_config={**CURRENT["text_config"], "extra": 1})

        check = compare_provenance(stored, CURRENT)

        assert check.status == "mismatch"
        assert "text_config.extra" in check.differences[0]


OK = IndexCheck("ok", [])
MISSING = IndexCheck("missing", [])
MISMATCH = IndexCheck("mismatch", ["model_name: index has 'a', config has 'b'"])


class TestPlanGenerate:
    @pytest.mark.parametrize("has_embeddings", [True, False])
    def test_ok_record_embeds_only_the_missing_rows(self, has_embeddings):
        assert plan_generate(OK, has_embeddings, rebuild_all=False, adopt=False) == "embed_missing"

    def test_no_record_and_no_embeddings_embeds(self):
        assert plan_generate(MISSING, False, rebuild_all=False, adopt=False) == "embed_missing"

    def test_all_re_embeds_in_every_state(self):
        for check in (OK, MISSING, MISMATCH):
            for has_embeddings in (True, False):
                assert plan_generate(check, has_embeddings, rebuild_all=True, adopt=False) == "embed_all"

    def test_mismatch_refuses_partial_embedding_and_names_the_differences_and_all(self):
        with pytest.raises(ProvenanceError) as exc:
            plan_generate(MISMATCH, True, rebuild_all=False, adopt=False)

        message = str(exc.value)
        assert "model_name: index has 'a', config has 'b'" in message
        assert "--all" in message

    def test_mismatch_refuses_even_when_no_embeddings_exist(self):
        with pytest.raises(ProvenanceError, match="--all"):
            plan_generate(MISMATCH, False, rebuild_all=False, adopt=False)

    def test_legacy_index_without_record_refuses_and_offers_adopt_and_all(self):
        with pytest.raises(ProvenanceError) as exc:
            plan_generate(MISSING, True, rebuild_all=False, adopt=False)

        assert "--adopt" in str(exc.value)
        assert "--all" in str(exc.value)

    def test_adopt_records_a_legacy_index(self):
        assert plan_generate(MISSING, True, rebuild_all=False, adopt=True) == "adopt"

    def test_adopt_with_nothing_to_adopt_is_refused(self):
        with pytest.raises(ProvenanceError, match="no embeddings"):
            plan_generate(MISSING, False, rebuild_all=False, adopt=True)

    @pytest.mark.parametrize("check", [OK, MISMATCH])
    def test_adopt_over_an_existing_record_is_refused(self, check):
        with pytest.raises(ProvenanceError, match="--all"):
            plan_generate(check, True, rebuild_all=False, adopt=True)

    def test_adopt_and_all_together_are_refused(self):
        with pytest.raises(ProvenanceError, match="mutually exclusive"):
            plan_generate(MISSING, True, rebuild_all=True, adopt=True)


class TestCheckVerdict:
    def test_ok(self):
        verdict = check_verdict(OK, has_embeddings=True)

        assert verdict.healthy is True
        assert verdict.level == "ok"

    def test_missing_without_embeddings_is_not_yet_run_and_healthy(self):
        verdict = check_verdict(MISSING, has_embeddings=False)

        assert verdict.healthy is True
        assert verdict.level == "notice"
        assert "generate_embeddings" in verdict.message

    def test_missing_with_embeddings_warns_about_legacy_and_stays_healthy(self):
        verdict = check_verdict(MISSING, has_embeddings=True)

        assert verdict.healthy is True
        assert verdict.level == "warning"
        assert "generate_embeddings --adopt" in verdict.message

    def test_mismatch_is_unhealthy_and_lists_differences_and_fix(self):
        verdict = check_verdict(MISMATCH, has_embeddings=True)

        assert verdict.healthy is False
        assert verdict.level == "error"
        assert "model_name: index has 'a', config has 'b'" in verdict.message
        assert "generate_embeddings --all" in verdict.message
