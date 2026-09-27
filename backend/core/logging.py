"""
Structured logging setup, shared across the whole application.

Usage:
    from backend.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("something happened", extra_field=123)
"""
import sys

from loguru import logger as _logger

from backend.core.config import get_settings

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    _configured = True


def get_logger(name: str):
    """Return a logger bound with a module name, configuring sinks on first use."""
    _configure()
    return _logger.bind(module=name)
