"""Central logger for the package."""
from __future__ import annotations

import logging
import os
import sys

_INITIALIZED = False


def _init_root_logger() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    level_name = os.environ.get("DOCUVISION_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("docuvision")
    logger.setLevel(level)
    # Only add handler if none (avoid duplicates under multi-import)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    _INITIALIZED = True


def get_logger(name: str = "docuvision") -> logging.Logger:
    """Return the DocuVision logger (or a child logger).

    Environment:
        DOCUVISION_LOG_LEVEL  - DEBUG / INFO / WARNING / ERROR (default INFO)
    """
    _init_root_logger()
    if name == "docuvision":
        return logging.getLogger("docuvision")
    if not name.startswith("docuvision."):
        name = f"docuvision.{name}"
    return logging.getLogger(name)
