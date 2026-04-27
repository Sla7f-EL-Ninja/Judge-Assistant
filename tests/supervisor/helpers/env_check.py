"""env_check.py — fail fast if required env vars are missing."""

import os

REQUIRED = [
    "MONGO_URI",
    "GOOGLE_API_KEY",
]

OPTIONAL_WITH_DEFAULTS = {
    "JA_QDRANT_HOST": "localhost",
    "JA_REDIS_URL": "redis://localhost:6379",
    "JA_MINIO_ENDPOINT": "localhost:9000",
    "JA_MONGODB_DATABASE": "judge_assistant_test",
}


def assert_env() -> None:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars for supervisor tests: {missing}\n"
            "Set them before running: export MONGO_URI=... GOOGLE_API_KEY=..."
        )


def get_test_db_name() -> str:
    return os.getenv("JA_MONGODB_DATABASE", "judge_assistant_test")


def get_mongo_uri() -> str:
    return os.getenv("MONGO_URI", "mongodb://localhost:27017/")


def get_qdrant_host() -> str:
    return os.getenv("JA_QDRANT_HOST", "localhost")


def get_qdrant_port() -> int:
    return int(os.getenv("JA_QDRANT_PORT", "6333"))


def get_redis_url() -> str:
    return os.getenv("JA_REDIS_URL", "redis://localhost:6379")


def get_minio_endpoint() -> str:
    return os.getenv("JA_MINIO_ENDPOINT", "localhost:9000")


def get_minio_access_key() -> str:
    return os.getenv("JA_MINIO_ACCESS_KEY", "minioadmin")


def get_minio_secret_key() -> str:
    return os.getenv("JA_MINIO_SECRET_KEY", "minioadmin")
