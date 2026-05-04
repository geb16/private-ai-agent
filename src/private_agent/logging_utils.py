"""Application logging utilities."""

from __future__ import annotations

import logging

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)


def configure_logging(log_level: str) -> None:
    """Configure global logging once per process."""
    logging.basicConfig(
        level=log_level,
        format=_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
