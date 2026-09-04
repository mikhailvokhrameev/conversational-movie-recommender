import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

SCORE_WEIGHTS = {
    "semantic": float(os.environ.get("SCORE_WEIGHT_SEMANTIC", "0.4")),
    "metadata": float(os.environ.get("SCORE_WEIGHT_METADATA", "0.3")),
    "session": float(os.environ.get("SCORE_WEIGHT_SESSION", "0.3")),
}

CATALOG_PARQUET_PATH = os.environ.get(
    "CATALOG_PARQUET_PATH", str(BASE_DIR / "catalog_okko.parquet")
)
