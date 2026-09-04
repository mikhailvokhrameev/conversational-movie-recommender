import os
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_params() -> dict:
    """Load params.yaml, the single source of truth for model and tuning values.

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
            with open(path) as handle:
                return yaml.safe_load(handle)
    raise FileNotFoundError(
        "params.yaml not found in: "
        + ", ".join(str(c) for c in candidates)
        + ". It holds every model and tuning parameter and is required."
    )


PARAMS = _load_params()

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

LANGUAGE_CODE = "ru-ru"
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
OLLAMA_BASE_URL = _llm["base_url"]
OLLAMA_MODEL = _llm["model"]
OLLAMA_TIMEOUTS = _llm["timeout_seconds"]
OLLAMA_THINKING = _llm["thinking"]

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

_session = PARAMS["session"]
SESSION_ALPHA = _session["alpha"]
SESSION_TTL_HOURS = _session["ttl_hours"]

GENRE_MATCH_THRESHOLD = PARAMS["intent"]["genre_match_threshold"]

_catalog = PARAMS["catalog"]
IMPORT_BATCH_SIZE = _catalog["import_batch_size"]
EMBEDDING_BATCH_SIZE = _catalog["embedding_batch_size"]

CATALOG_PARQUET_PATH = os.environ.get(
    "CATALOG_PARQUET_PATH", str(BASE_DIR / "catalog_okko.parquet")
)
