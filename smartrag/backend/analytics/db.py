"""
SQLite engine/session setup for request logging.

Kept as a thin, swappable layer: everything downstream talks to `get_session()`
and SQLModel objects, so switching `sqlite:///...` for a Postgres URL in
Settings is the only change needed to move to Postgres later.
"""
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from backend.core.config import Settings, get_settings

_engine = None


def get_engine(settings: Settings | None = None):
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        db_path = Path(settings.sqlite_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(_engine)
    return _engine


def get_session(settings: Settings | None = None) -> Session:
    return Session(get_engine(settings))


def reset_engine_for_tests() -> None:
    """Test-only helper to force a fresh engine (e.g. after pointing
    SQLITE_DB_PATH at a temp file)."""
    global _engine
    _engine = None
