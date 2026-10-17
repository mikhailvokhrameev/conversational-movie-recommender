import os
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def _find_params() -> Path:
    """Locate params.yaml, the single source of truth for model and tuning values.

    Searched in the container location first (mounted at /app/params.yaml),
    then the repo root for running outside Docker. PARAMS_PATH overrides both;
    it is deployment wiring, not a tuning knob.
    """
    candidates = [Path(os.environ["PARAMS_PATH"])] if os.environ.get("PARAMS_PATH") else [
        BASE_DIR / "params.yaml",
        BASE_DIR.parent / "params.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "params.yaml not found in: "
        + ", ".join(str(c) for c in candidates)
        + ". It holds every model and tuning parameter and is required."
    )


# The resolved path is kept so evaluation run records can hash the exact file
# the run was configured from.
PARAMS_FILE = _find_params()
with open(PARAMS_FILE) as _params_handle:
    PARAMS = yaml.safe_load(_params_handle)

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-key-do-not-use-in-prod")
DEBUG = os.environ.get("DEBUG", "0") == "1"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "rest_framework",
    "corsheaders",
    "movies",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "recommender.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

ASGI_APPLICATION = "recommender.asgi.application"

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgres://recommender:changeme@localhost:5432/recommender"
)

_db_parts = DATABASE_URL.replace("postgres://", "").split("@")
_user_pass = _db_parts[0].split(":")
_host_db = _db_parts[1].split("/")
_host_port = _host_db[0].split(":")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _host_db[1] if len(_host_db) > 1 else "recommender",
        "USER": _user_pass[0],
        "PASSWORD": _user_pass[1] if len(_user_pass) > 1 else "",
        "HOST": _host_port[0],
        "PORT": _host_port[1] if len(_host_port) > 1 else "5432",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"

CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
}

# --- Values below come from params.yaml. Do not add os.environ fallbacks: the
# --- point of that file is that it is the only place these can be changed.

_llm = PARAMS["llm"]


def _validate_num_ctx(llm: dict) -> None:
    """Fail at load if llm.num_ctx is not a positive integer.

    It is sent as options.num_ctx on every LLM call; a non-int or non-positive
    value would reach Ollama and fail late or be silently mishandled.
    """
    value = llm["num_ctx"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"params.yaml: llm.num_ctx must be an integer >= 1 (got {value!r})."
        )


def _validate_sampling(llm: dict) -> None:
    """Fail at load if llm.temperature / llm.seed are malformed.

    Both kinds of temperature must be present and numeric in [0, 2] (above 2 sampling
    is effectively noise); seed must be an int. A bad value would otherwise reach
    Ollama on every call and be rejected or silently ignored there.
    """
    temperature = llm["temperature"]
    if not isinstance(temperature, dict):
        raise ValueError(
            "params.yaml: llm.temperature must map structured and generation "
            f"to a temperature (got {temperature!r})."
        )
    for kind in ("structured", "generation"):
        value = temperature.get(kind)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
            raise ValueError(
                f"params.yaml: llm.temperature.{kind} must be a number in [0, 2] (got {value!r})."
            )
    seed = llm["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"params.yaml: llm.seed must be an integer (got {seed!r}).")


def _validate_explanation_description_chars(llm: dict) -> None:
    """Fail at load if llm.explanation_description_chars is not a positive integer.

    0 would hand the explanation prompt titles only, and the model would then
    describe films from memory instead of from the catalog.
    """
    value = llm["explanation_description_chars"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"params.yaml: llm.explanation_description_chars must be an integer >= 1 (got {value!r})."
        )


def _validate_placeholder_max_chars(llm: dict) -> None:
    """Fail at load if llm.explanation.placeholder_max_chars is not a positive
    integer: it is the longest <...> span the explanation stream treats as an
    echoed prompt placeholder, and 0 would let every one through."""
    explanation = llm.get("explanation")
    value = explanation.get("placeholder_max_chars") if isinstance(explanation, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"params.yaml: llm.explanation.placeholder_max_chars must be an integer >= 1 (got {value!r})."
        )


_validate_num_ctx(_llm)
_validate_sampling(_llm)
_validate_explanation_description_chars(_llm)
OLLAMA_BASE_URL = _llm["base_url"]
OLLAMA_MODEL = _llm["model"]
OLLAMA_TIMEOUTS = _llm["timeout_seconds"]
OLLAMA_THINKING = _llm["thinking"]
OLLAMA_NUM_CTX = _llm["num_ctx"]
OLLAMA_TEMPERATURE = _llm["temperature"]
OLLAMA_SEED = _llm["seed"]
OLLAMA_EXPLANATION_DESCRIPTION_CHARS = _llm["explanation_description_chars"]
_validate_placeholder_max_chars(_llm)
OLLAMA_PLACEHOLDER_MAX_CHARS = _llm["explanation"]["placeholder_max_chars"]

def _validate_embedding_text(embedding: dict) -> None:
    """Fail at load if embedding.text.* is not a pair of positive integers.

    Both feed the text that gets embedded and are recorded in the index
    provenance. 0 would embed titles with no description or no keywords, and a
    non-int would only fail (or silently slice wrong) inside generate_embeddings.
    """
    text = embedding.get("text")
    if not isinstance(text, dict):
        raise ValueError(
            "params.yaml: embedding.text must map description_chars and max_keywords "
            f"to positive integers (got {text!r})."
        )
    for key in ("description_chars", "max_keywords"):
        value = text.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"params.yaml: embedding.text.{key} must be an integer >= 1 (got {value!r})."
            )


def _validate_revision(section: str, params: dict) -> None:
    """Fail at load if <section>.revision is neither null nor a non-empty string.

    It is passed straight to Hugging Face as the `revision` of the model
    download; null means "whatever the default branch is today" (unpinned).
    """
    value = params["revision"]
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(
            f"params.yaml: {section}.revision must be null (unpinned) or a non-empty "
            f"commit hash string (got {value!r})."
        )


_embedding = PARAMS["embedding"]
_validate_embedding_text(_embedding)
_validate_revision("embedding", _embedding)
EMBEDDING_MODEL = _embedding["model"]
EMBEDDING_REVISION = _embedding["revision"]
EMBEDDING_DIMENSIONS = _embedding["dimensions"]
EMBEDDING_TEXT_DESCRIPTION_CHARS = _embedding["text"]["description_chars"]
EMBEDDING_TEXT_MAX_KEYWORDS = _embedding["text"]["max_keywords"]

_retrieval = PARAMS["retrieval"]
CANDIDATE_COUNT = _retrieval["candidate_count"]
RRF_K = _retrieval["rrf_k"]
RRF_WEIGHTS = _retrieval["rrf_weights"]
LEXICAL_SEARCH_CONFIG = _retrieval["lexical"]["text_search_config"]
LEXICAL_MIN_TERM_LENGTH = _retrieval["lexical"]["min_term_length"]
LEXICAL_MAX_TERMS = _retrieval["lexical"]["max_terms"]

def _validate_score_weights(weights: dict) -> None:
    """Fail at load if scoring.weights has a negative entry or does not sum to 1.

    The final score is a weighted sum of min-max normalized signals, so weights
    that do not sum to 1.0 silently rescale every score (and with it the
    meaning of reranking.weight and the rerank gate), and a negative weight
    would reward the opposite of the signal.
    """
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"params.yaml: scoring.weights must map signal names to weights (got {weights!r}).")
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(
                f"params.yaml: scoring.weights.{name} must be a number >= 0 (got {value!r})."
            )
    # Literal: settings cannot import core.scoring.SIGNALS.
    if set(weights) != {"semantic", "metadata", "popularity"}:
        raise ValueError(
            "params.yaml: scoring.weights must have exactly the signals "
            f"semantic, metadata, popularity (got {sorted(weights)})."
        )
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"params.yaml: scoring.weights must sum to 1.0 (got {total!r} from {weights!r})."
        )


_scoring = PARAMS["scoring"]
_validate_score_weights(_scoring["weights"])
SCORE_WEIGHTS = _scoring["weights"]
NEUTRAL_SCORE = _scoring["neutral_score"]

_diversification = PARAMS["diversification"]
TOP_N = _diversification["top_n"]
MMR_LAMBDA = _diversification["lambda"]

def _validate_rerank_min_score(reranking: dict) -> None:
    """Fail at load if reranking.min_score is neither null nor inside (0, 1).

    rerank_score is a sigmoid output in [0, 1] (see the params.yaml comment).
    0 gates nothing and 1 gates everything, and a typo like 5 or 30 (a logit-
    scale habit) would silently empty every answer, so both fail loudly.
    """
    value = reranking["min_score"]
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError(
            f"params.yaml: reranking.min_score must be null (gate off) or a number strictly "
            f"between 0 and 1, the range of the sigmoid-scaled cross-encoder score (got {value!r})."
        )


def _validate_rerank_weight(reranking: dict) -> None:
    """Fail at load if reranking.weight is not a number in [0, 1].

    It is the share of the final score taken from the cross-encoder; the scorer
    gets the remaining 1 - weight, so a value outside [0, 1] would give one side
    a negative share.
    """
    value = reranking["weight"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"params.yaml: reranking.weight must be a number in [0, 1] (got {value!r}).")


def _validate_lexical_stopwords(stopwords: list) -> None:
    """Fail at load if retrieval.lexical.stopwords is not a list of non-empty
    lowercase strings -- terms are lowercased before the lookup, so an
    uppercase entry could never match."""
    if not isinstance(stopwords, list) or not all(
        isinstance(w, str) and w and w == w.lower() for w in stopwords
    ):
        raise ValueError(
            "params.yaml: retrieval.lexical.stopwords must be a list of non-empty "
            f"lowercase strings (got {stopwords!r})."
        )


_validate_lexical_stopwords(_retrieval["lexical"]["stopwords"])
LEXICAL_STOPWORDS = frozenset(_retrieval["lexical"]["stopwords"])

_reranking = PARAMS["reranking"]
_validate_rerank_min_score(_reranking)
_validate_rerank_weight(_reranking)
_validate_revision("reranking", _reranking)
RERANKER_ENABLED = _reranking["enabled"]
RERANKER_MODEL = _reranking["model"]
RERANKER_REVISION = _reranking["revision"]
RERANKER_DEVICE = _reranking["device"]
RERANK_TOP_K = _reranking["top_k"]
RERANK_WEIGHT = _reranking["weight"]
RERANK_MIN_SCORE = _reranking["min_score"]
RERANK_MAX_LENGTH = _reranking["max_length"]
RERANK_DESCRIPTION_CHARS = _reranking["description_chars"]

def _validate_history_limits(session: dict) -> None:
    """Fail at load if a history limit is < 1.

    0 must not mean "unlimited": the history window is what keeps the prompt
    inside llm.num_ctx, so a non-positive limit would silently remove that bound.
    """
    for key in (
        "history_max_turns", "history_max_chars_per_entry", "history_movie_description_chars",
    ):
        value = session[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"params.yaml: session.{key} must be an integer >= 1 (got {value!r}). "
                "0 does not mean unlimited -- the history window bounds the prompt "
                "to llm.num_ctx."
            )


_session = PARAMS["session"]
_validate_history_limits(_session)
SESSION_TTL_HOURS = _session["ttl_hours"]
SESSION_HISTORY_MAX_TURNS = _session["history_max_turns"]
SESSION_HISTORY_MAX_CHARS_PER_ENTRY = _session["history_max_chars_per_entry"]
SESSION_HISTORY_MOVIE_DESCRIPTION_CHARS = _session["history_movie_description_chars"]


# Which resolution fields get the embedding fallback (core/entity_resolution.py).
# languages does not: ISO codes mean nothing to an embedder, so it takes
# synonyms only.
_RESOLUTION_EMBEDDED = {"genres": True, "countries": True, "languages": False}


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _validate_intent(intent) -> None:
    """Fail at load if intent.resolution is malformed or the removed
    intent.genre_match_threshold is still set.

    Each field maps short synonyms (name -> catalog value) and, where the
    embedding fallback applies, match_threshold in (0, 1] and match_margin in
    [0, 1): a top-1 cosine below the threshold, or within the margin of the
    runner-up, drops the value instead of guessing.
    """
    if isinstance(intent, dict) and "genre_match_threshold" in intent:
        raise ValueError(
            "params.yaml: intent.genre_match_threshold was removed; genre matching is "
            "configured under intent.resolution.genres (match_threshold, match_margin, synonyms)."
        )
    resolution = intent.get("resolution") if isinstance(intent, dict) else None
    if not isinstance(resolution, dict) or set(resolution) != set(_RESOLUTION_EMBEDDED):
        raise ValueError(
            "params.yaml: intent.resolution must have exactly the fields "
            f"{', '.join(_RESOLUTION_EMBEDDED)} (got {intent!r})."
        )
    for field, embedded in _RESOLUTION_EMBEDDED.items():
        prefix = f"params.yaml: intent.resolution.{field}"
        config = resolution[field]
        expected = {"synonyms", "match_threshold", "match_margin"} if embedded else {"synonyms"}
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError(f"{prefix} must have exactly the keys {', '.join(sorted(expected))} (got {config!r}).")
        synonyms = config["synonyms"]
        if not isinstance(synonyms, dict) or not all(
            isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
            for k, v in synonyms.items()
        ):
            raise ValueError(
                f"{prefix}.synonyms must map non-blank names to non-blank catalog values; quote "
                f"YAML words like no/yes/on/off (got {synonyms!r})."
            )
        if not embedded:
            continue
        threshold = config["match_threshold"]
        if not _is_number(threshold) or not 0 < threshold <= 1:
            raise ValueError(f"{prefix}.match_threshold must be a number in (0, 1] (got {threshold!r}).")
        margin = config["match_margin"]
        if not _is_number(margin) or not 0 <= margin < 1:
            raise ValueError(f"{prefix}.match_margin must be a number in [0, 1) (got {margin!r}).")


_validate_intent(PARAMS["intent"])
INTENT_RESOLUTION = PARAMS["intent"]["resolution"]


def _validate_evaluation(evaluation: dict) -> None:
    """Fail at load if an evaluation.* judge value is not positive.

    A zero description cap would hand the judge titles only, and a zero
    timeout would turn every judge call into a counted failure.
    """
    chars = evaluation["judge_description_chars"]
    if isinstance(chars, bool) or not isinstance(chars, int) or chars < 1:
        raise ValueError(
            f"params.yaml: evaluation.judge_description_chars must be an integer >= 1 (got {chars!r})."
        )
    timeout = evaluation["judge_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(
            f"params.yaml: evaluation.judge_timeout_seconds must be a number > 0 (got {timeout!r})."
        )


_evaluation = PARAMS["evaluation"]
_validate_evaluation(_evaluation)
EVAL_JUDGE_DESCRIPTION_CHARS = _evaluation["judge_description_chars"]
EVAL_JUDGE_TIMEOUT_SECONDS = _evaluation["judge_timeout_seconds"]

def _validate_unchecked_attributes(catalog: dict) -> None:
    """Fail at load if catalog.unchecked_attributes is not a non-empty list of
    distinct, non-blank strings -- it is rendered into the classify prompt and
    the "can't check" reply."""
    value = catalog.get("unchecked_attributes")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(a, str) and a.strip() for a in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "params.yaml: catalog.unchecked_attributes must be a non-empty list of "
            f"distinct non-blank strings (got {value!r})."
        )


def _unchecked_text(attributes: list) -> str:
    """'a, b and c' -- the one wording both the prompt and the reply use."""
    if len(attributes) == 1:
        return attributes[0]
    return ", ".join(attributes[:-1]) + " and " + attributes[-1]


_catalog = PARAMS["catalog"]
_validate_unchecked_attributes(_catalog)
CATALOG_UNCHECKED_ATTRIBUTES = tuple(_catalog["unchecked_attributes"])
CATALOG_UNCHECKED_TEXT = _unchecked_text(_catalog["unchecked_attributes"])
IMPORT_BATCH_SIZE = _catalog["import_batch_size"]
EMBEDDING_BATCH_SIZE = _catalog["embedding_batch_size"]
MIN_VOTE_COUNT = _catalog["min_vote_count"]
TMDB_POSTER_BASE_URL = _catalog["poster_base_url"]
TMDB_POSTER_SIZE = _catalog["poster_size"]

CATALOG_CSV_PATH = os.environ.get(
    "CATALOG_CSV_PATH", str(BASE_DIR / "TMDB Movie Dataset v11.csv")
)

def _validate_logging_level(logging_params) -> None:
    """Fail at load if logging.level is not an upper-case stdlib level name."""
    value = logging_params.get("level") if isinstance(logging_params, dict) else None
    if value not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(
            "params.yaml: logging.level must be one of DEBUG, INFO, WARNING, ERROR, "
            f"CRITICAL (got {value!r})."
        )


_validate_logging_level(PARAMS.get("logging"))
LOG_LEVEL = PARAMS["logging"]["level"]

# The app's own loggers log per-call stats (`Ollama call=...`) and per-turn
# latency at INFO (logging.level in params.yaml). Without this, Python's last-resort handler drops everything
# below WARNING and those lines never reach `docker compose logs backend`.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "plain"},
    },
    "loggers": {
        "core": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": True},
        "movies": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": True},
    },
}
