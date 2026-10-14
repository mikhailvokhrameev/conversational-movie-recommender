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


_validate_num_ctx(_llm)
_validate_sampling(_llm)
OLLAMA_BASE_URL = _llm["base_url"]
OLLAMA_MODEL = _llm["model"]
OLLAMA_TIMEOUTS = _llm["timeout_seconds"]
OLLAMA_THINKING = _llm["thinking"]
OLLAMA_NUM_CTX = _llm["num_ctx"]
OLLAMA_TEMPERATURE = _llm["temperature"]
OLLAMA_SEED = _llm["seed"]

_embedding = PARAMS["embedding"]
EMBEDDING_MODEL = _embedding["model"]
EMBEDDING_DIMENSIONS = _embedding["dimensions"]

_retrieval = PARAMS["retrieval"]
CANDIDATE_COUNT = _retrieval["candidate_count"]
RRF_K = _retrieval["rrf_k"]
RRF_WEIGHTS = _retrieval["rrf_weights"]
LEXICAL_SEARCH_CONFIG = _retrieval["lexical"]["text_search_config"]
LEXICAL_MIN_TERM_LENGTH = _retrieval["lexical"]["min_term_length"]
LEXICAL_MAX_TERMS = _retrieval["lexical"]["max_terms"]

_scoring = PARAMS["scoring"]
SCORE_WEIGHTS = _scoring["weights"]
NEUTRAL_SCORE = _scoring["neutral_score"]

_diversification = PARAMS["diversification"]
TOP_N = _diversification["top_n"]
MMR_LAMBDA = _diversification["lambda"]

_reranking = PARAMS["reranking"]
RERANKER_ENABLED = _reranking["enabled"]
RERANKER_MODEL = _reranking["model"]
RERANKER_DEVICE = _reranking["device"]
RERANK_TOP_K = _reranking["top_k"]
RERANK_WEIGHT = _reranking["weight"]
RERANK_MAX_LENGTH = _reranking["max_length"]
RERANK_DESCRIPTION_CHARS = _reranking["description_chars"]

def _validate_history_limits(session: dict) -> None:
    """Fail at load if a history limit is < 1.

    0 must not mean "unlimited": the history window is what keeps the prompt
    inside llm.num_ctx, so a non-positive limit would silently remove that bound.
    """
    for key in ("history_max_turns", "history_max_chars_per_entry"):
        value = session[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"params.yaml: session.{key} must be an integer >= 1 (got {value!r}). "
                "0 does not mean unlimited -- the history window bounds the prompt "
                "to llm.num_ctx."
            )


_session = PARAMS["session"]
_validate_history_limits(_session)
SESSION_ALPHA = _session["alpha"]
SESSION_ALPHA_REFINEMENT = _session["alpha_refinement"]
SESSION_TTL_HOURS = _session["ttl_hours"]
SESSION_HISTORY_MAX_TURNS = _session["history_max_turns"]
SESSION_HISTORY_MAX_CHARS_PER_ENTRY = _session["history_max_chars_per_entry"]

GENRE_MATCH_THRESHOLD = PARAMS["intent"]["genre_match_threshold"]


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

_catalog = PARAMS["catalog"]
IMPORT_BATCH_SIZE = _catalog["import_batch_size"]
EMBEDDING_BATCH_SIZE = _catalog["embedding_batch_size"]
MIN_VOTE_COUNT = _catalog["min_vote_count"]
TMDB_POSTER_BASE_URL = _catalog["poster_base_url"]
TMDB_POSTER_SIZE = _catalog["poster_size"]

CATALOG_CSV_PATH = os.environ.get(
    "CATALOG_CSV_PATH", str(BASE_DIR / "TMDB Movie Dataset v11.csv")
)
