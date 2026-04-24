"""
config -- Centralized configuration for Judge Assistant.

Loads ``settings.yaml`` as defaults, deep-merges ``settings.local.yaml``
(if present), and applies environment variable overrides using a ``JA_``
prefix convention.

Public API
----------
- ``cfg``      -- the loaded :class:`AppConfig` singleton
- ``get_llm``  -- factory that returns a LangChain chat model for a tier

Environment Variable Override Convention
----------------------------------------
Nested YAML keys are joined with ``_`` and upper-cased, prefixed with ``JA_``.
Examples::

    JA_LLM_HIGH_MODEL=gpt-4o
    JA_MONGODB_URI=mongodb+srv://...
    JA_OCR_LANGUAGE=en

API keys (``GROQ_API_KEY``, ``GOOGLE_API_KEY``) are **not** stored in YAML.
They stay in ``.env`` or the process environment and are read directly by
the LangChain provider classes.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Used to prevent callers from mutating returned section dicts (P1.1.7)
_EMPTY: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML = _CONFIG_DIR / "settings.yaml"
_LOCAL_YAML = _CONFIG_DIR / "settings.local.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_env_overrides(data: Dict[str, Any], prefix: str = "JA") -> Dict[str, Any]:
    """Override YAML values with environment variables.

    The convention is ``JA_<SECTION>_<KEY>`` for top-level keys and
    ``JA_<SECTION>_<SUB>_<KEY>`` for nested keys.  Values are cast to
    match the type of the YAML default (bool, int, float, str).

    Net-new keys absent from YAML can also be injected via env (P1.9.5/B27):
    ``JA_SECTION_KEY=value`` injects ``{section: {key: value}}`` as a string
    if no matching path exists in the loaded config.
    """
    prefix_upper = prefix.upper() + "_"
    flat = _flatten(data)
    covered_env_keys: set = set()

    # Pass 1 — update existing YAML keys
    for flat_key, default_value in flat.items():
        env_key = f"{prefix}_{'_'.join(flat_key)}".upper()
        covered_env_keys.add(env_key)
        env_val = os.environ.get(env_key)
        if env_val is not None:
            cast_val = _cast(env_val, default_value)
            _set_nested(data, flat_key, cast_val)

    # Pass 2 — inject net-new keys absent from YAML (P1.9.5)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix_upper):
            continue
        if env_key in covered_env_keys:
            continue
        # Derive key path: JA_SECTION_KEY → ("section", "key")
        remainder = env_key[len(prefix_upper):]
        parts = tuple(p.lower() for p in remainder.split("_") if p)
        if len(parts) >= 2:
            _set_nested(data, parts, env_val)

    return data


def _flatten(
    d: Dict[str, Any], parent_keys: tuple = ()
) -> Dict[tuple, Any]:
    """Flatten a nested dict into ``{(key, subkey, ...): leaf_value}``."""
    items: Dict[tuple, Any] = {}
    for k, v in d.items():
        new_key = parent_keys + (k,)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        else:
            items[new_key] = v
    return items


def _set_nested(d: Dict[str, Any], keys: tuple, value: Any) -> None:
    """Set a value in a nested dict following the key path."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _cast(value: str, reference: Any) -> Any:
    """Cast a string *value* to the type of *reference*."""
    if isinstance(reference, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, list):
        # Comma-separated string -> list
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------

class AppConfig:
    """Typed, dict-backed configuration singleton.

    Access sections as attributes::

        cfg.llm           # {'high': {...}, 'medium': {...}, 'low': {...}}
        cfg.mongodb       # {'uri': '...', 'database': '...', ...}
        cfg.api           # {'app_name': '...', ...}
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    # -- dict-style access ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("AppConfig is read-only after initialization — use settings.local.yaml or JA_* env vars")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # -- attribute access for top-level sections (P1.1.7) --------------------
    # Each property returns a shallow copy so callers cannot mutate the
    # shared singleton's internal state via the returned dict.

    def _section(self, key: str) -> Dict[str, Any]:
        return dict(self._data.get(key, _EMPTY))

    @property
    def llm(self) -> Dict[str, Any]:
        return self._section("llm")

    @property
    def embedding(self) -> Dict[str, Any]:
        return self._section("embedding")

    @property
    def mongodb(self) -> Dict[str, Any]:
        return self._section("mongodb")

    @property
    def qdrant(self) -> Dict[str, Any]:
        return self._section("qdrant")

    @property
    def redis(self) -> Dict[str, Any]:
        return self._section("redis")

    @property
    def minio(self) -> Dict[str, Any]:
        return self._section("minio")

    @property
    def postgresql(self) -> Dict[str, Any]:
        return self._section("postgresql")

    @property
    def api(self) -> Dict[str, Any]:
        return self._section("api")

    @property
    def supervisor(self) -> Dict[str, Any]:
        return self._section("supervisor")

    @property
    def rag(self) -> Dict[str, Any]:
        return self._section("rag")

    @property
    def ocr(self) -> Dict[str, Any]:
        return self._section("ocr")

    @property
    def tei(self) -> Dict[str, Any]:
        return self._section("tei")

    def raw(self) -> Dict[str, Any]:
        """Return the full raw config dict."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        sections = ", ".join(self._data.keys())
        return f"<AppConfig sections=[{sections}]>"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_config() -> AppConfig:
    """Load settings.yaml, merge local overrides, apply env vars."""
    if not _DEFAULT_YAML.exists():
        raise FileNotFoundError(
            f"Default configuration file not found: {_DEFAULT_YAML}"
        )

    with open(_DEFAULT_YAML, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Merge local overrides
    if _LOCAL_YAML.exists():
        with open(_LOCAL_YAML, "r", encoding="utf-8") as fh:
            local_data = yaml.safe_load(fh) or {}
        data = _deep_merge(data, local_data)
        logger.info("Merged local config overrides from %s", _LOCAL_YAML)

    # Apply environment variable overrides
    data = _apply_env_overrides(data)

    return AppConfig(data)


# Module-level singleton -- created on first import
cfg: AppConfig = _load_config()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

_VALID_TIERS = ("high", "medium", "low")


def get_llm(tier: str, **overrides: Any):
    """Return a LangChain chat model for the requested complexity tier.

    Parameters
    ----------
    tier : str
        One of ``"high"``, ``"medium"``, ``"low"``.
    **overrides
        Keyword arguments forwarded to the provider constructor,
        e.g. ``temperature=0.3``.

    Returns
    -------
    langchain_core.language_models.BaseChatModel
        A ready-to-use LangChain chat model instance.

    Raises
    ------
    ValueError
        If *tier* is not one of the valid tiers.
    ValueError
        If the configured provider is not supported.
    """
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Unknown LLM tier '{tier}'. Must be one of {_VALID_TIERS}."
        )

    tier_cfg = cfg.llm.get(tier, {})
    provider = tier_cfg.get("provider", "google")
    model = tier_cfg.get("model", "gemini-2.5-flash")
    temperature = tier_cfg.get("temperature", 0.0)

    # Allow caller overrides
    model = overrides.pop("model", model)
    temperature = overrides.pop("temperature", temperature)
    # Default 120 s per call; callers may override via request_timeout=N
    timeout = overrides.pop("request_timeout", tier_cfg.get("timeout_seconds", 120))

    llm_retries = overrides.pop("max_retries", tier_cfg.get("max_retries", 3))

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model_name=model,
            temperature=temperature,
            request_timeout=timeout,
            max_retries=llm_retries,
            **overrides,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            request_timeout=timeout,
            max_retries=llm_retries,
            **overrides,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}' in tier '{tier}'. "
        "Supported providers: groq, google."
    )
