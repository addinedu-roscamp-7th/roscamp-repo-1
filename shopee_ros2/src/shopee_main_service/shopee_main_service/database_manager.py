"""
데이터베이스 관리자

ORM(SQLAlchemy) 세션 라이프사이클을 관리합니다.
- 세션 생성/커밋/롤백/종료 자동화
- 트랜잭션 관리
"""
from __future__ import annotations

import contextlib
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings


class DatabaseManager:
    """
    데이터베이스 세션 관리자
    
    SQLAlchemy ORM 세션의 생명주기를 관리합니다.
    - 세션 팩토리 제공
    - 자동 커밋/롤백
    - 리소스 정리
    """

    def __init__(self) -> None:
        """DB 엔진과 세션 팩토리를 초기화합니다."""
        self._engine = create_engine(
            settings.DB_URL,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            echo=settings.DB_ECHO,
        )
        self._session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine
        )

    @contextlib.contextmanager
    def session_scope(self) -> Iterator[Session]:
        """
        세션 컨텍스트 매니저
        
        with 문과 함께 사용하여 자동으로 세션을 관리합니다.
        - 정상 종료: commit
        - 예외 발생: rollback
        - 항상: close
        
        사용 예:
            with db_manager.session_scope() as session:
                customer = session.query(Customer).filter_by(id=user_id).first()
        
        Yields:
            Session: SQLAlchemy DB 세션 객체
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()  # 성공 시 커밋
        except Exception:
            session.rollback()  # 실패 시 롤백
            raise
        finally:
            session.close()  # 항상 종료

    def get_pool_stats(self) -> dict:
        """
        커넥션 풀 상태를 조회한다.

        Returns:
            사용 중 커넥션 수와 풀 크기를 포함한 딕셔너리
        """
        pool = self._engine.pool
        checked_out = 0
        if hasattr(pool, 'checkedout'):
            checked = pool.checkedout()
            if isinstance(checked, int):
                checked_out = checked
        pool_size = settings.DB_POOL_SIZE
        if hasattr(pool, 'size'):
            size_value = pool.size()
            if isinstance(size_value, int):
                pool_size = size_value
        return {
            'db_connections': checked_out,
            'db_connections_max': pool_size,
        }
