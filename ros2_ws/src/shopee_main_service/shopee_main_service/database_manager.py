from __future__ import annotations

import contextlib
from typing import Iterator


class DatabaseSession:
    """Placeholder ORM session wrapper."""

    def commit(self) -> None:  # pragma: no cover - placeholder
        pass

    def rollback(self) -> None:  # pragma: no cover - placeholder
        pass

    def close(self) -> None:  # pragma: no cover - placeholder
        pass


class DatabaseManager:
    """Provides lifecycle helpers around ORM sessions."""

    def __init__(self) -> None:
        self._engine = None  # actual engine to be injected later

    def configure(self, engine: object) -> None:
        self._engine = engine

    @contextlib.contextmanager
    def session_scope(self) -> Iterator[DatabaseSession]:
        session = DatabaseSession()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
