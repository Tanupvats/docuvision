"""Lazy optional-import helpers.

All heavy/optional dependencies are imported through these helpers so that a
missing library never breaks the package — it just disables a specific engine.
"""
from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=None)
def is_available(module_name: str) -> bool:
    """Return True if `module_name` can be imported. Result is cached."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return False
    return spec is not None


def optional_import(module_name: str) -> Optional[Any]:
    """Import a module, or return None if it is not installed.

    Never raises. Use this for optional engine backends.
    """
    if not is_available(module_name):
        return None
    try:
        return importlib.import_module(module_name)
    except Exception:
        # Broad except: some libraries raise at import time under certain
        # environments (e.g. missing shared libs). Treat those as unavailable.
        return None


def require(module_name: str, feature: str = "") -> Any:
    """Import a module or raise ImportError with a friendly message."""
    mod = optional_import(module_name)
    if mod is None:
        hint = f" (required for {feature})" if feature else ""
        raise ImportError(
            f"'{module_name}' is not installed{hint}. "
            "See https://github.com/your-org/docuvision#install for extras."
        )
    return mod
