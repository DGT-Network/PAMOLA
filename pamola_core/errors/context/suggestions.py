"""Recovery suggestions and contextual help for errors."""

from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import yaml  # type: ignore[import-untyped]


class ErrorContext:
    """Provides context and recovery suggestions for errors."""

    _DATA_FILE: ClassVar[Path] = Path(__file__).with_name("recovery_data.yaml")

    _RECOVERY_SUGGESTIONS: ClassVar[Dict[str, List[str]]] = {}
    _CATEGORY_SUGGESTIONS: ClassVar[Dict[str, List[str]]] = {}
    _DEFAULT_SUGGESTIONS: ClassVar[List[str]] = []

    @classmethod
    @lru_cache(maxsize=1)
    def _load_data(cls) -> Dict[str, Any]:
        """Load suggestions from YAML file (cached)."""
        with cls._DATA_FILE.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Populate class-level caches from YAML on first access."""
        if cls._RECOVERY_SUGGESTIONS:
            return

        data = cls._load_data()
        suggestions = data.get("suggestions") or {}
        category_suggestions = data.get("category_suggestions") or {}
        defaults = data.get("default") or []

        cls._RECOVERY_SUGGESTIONS = {
            str(code): [str(item) for item in items]
            for code, items in suggestions.items()
            if isinstance(items, list)
        }
        cls._CATEGORY_SUGGESTIONS = {
            str(category): [str(item) for item in items]
            for category, items in category_suggestions.items()
            if isinstance(items, list)
        }
        cls._DEFAULT_SUGGESTIONS = [str(item) for item in defaults]

    @classmethod
    def get_suggestions(cls, error_code: str) -> List[str]:
        """Get recovery suggestions for an error code."""
        cls._ensure_loaded()
        return cls._RECOVERY_SUGGESTIONS.get(error_code, cls._DEFAULT_SUGGESTIONS)

    @classmethod
    def get_category_suggestions(cls, category: str) -> List[str]:
        """Get general suggestions for an error category."""
        cls._ensure_loaded()
        return cls._CATEGORY_SUGGESTIONS.get(category.lower(), cls._DEFAULT_SUGGESTIONS)

    @classmethod
    def format_suggestions(
        cls,
        error_code: str,
        max_suggestions: Optional[int] = None,
    ) -> str:
        """Format recovery suggestions as numbered list."""
        suggestions = cls.get_suggestions(error_code)
        if max_suggestions:
            suggestions = suggestions[:max_suggestions]

        if not suggestions:
            return "No specific suggestions available."

        formatted = ["Recovery suggestions:"]
        for idx, suggestion in enumerate(suggestions, 1):
            formatted.append(f"  {idx}. {suggestion}")

        return "\n".join(formatted)

    @classmethod
    def has_suggestions(cls, error_code: str) -> bool:
        """Check if specific suggestions exist for an error code."""
        cls._ensure_loaded()
        return error_code in cls._RECOVERY_SUGGESTIONS


