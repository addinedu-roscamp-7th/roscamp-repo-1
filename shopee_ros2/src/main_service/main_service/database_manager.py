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
        
        # DB 활동 추적용 카운터
        self._active_sessions = 0
        self._total_sessions_created = 0

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
        self._active_sessions += 1
        self._total_sessions_created += 1
        
        # 디버깅용 로그 추가
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"DB session created. Active sessions: {self._active_sessions}")
        
        try:
            yield session
            session.commit()  # 성공 시 커밋
        except Exception as e:
            session.rollback()  # 실패 시 롤백
            logger.error(f"Database transaction failed and rolled back: {type(e).__name__}: {e}")
            raise
        finally:
            session.close()  # 항상 종료
            self._active_sessions -= 1
        logger.debug(f"DB session closed. Active sessions: {self._active_sessions}")

    def get_pool_stats(self) -> dict:
        """
        커넥션 풀 상태를 조회한다.

        Returns:
            사용 중 커넥션 수와 풀 크기를 포함한 딕셔너리
        """
        pool = self._engine.pool
        
        # 실제 커넥션 풀 상태 조회
        checked_out = 0
        pool_size = settings.DB_POOL_SIZE
        
        try:
            # SQLAlchemy 풀에서 실제 상태 조회
            if hasattr(pool, 'checkedout'):
                checked_out = pool.checkedout()
            else:
                # 풀에서 직접 조회할 수 없는 경우 활성 세션 수 사용
                checked_out = self._active_sessions
            
            # 실제 풀 크기 조회
            if hasattr(pool, 'size'):
                pool_size = pool.size()
            elif hasattr(pool, '_pool') and hasattr(pool._pool, 'queue'):
                # 큐에 있는 커넥션 수 + 체크아웃된 커넥션 수
                pool_size = len(pool._pool.queue) + checked_out
            else:
                # 설정값 사용
                pool_size = settings.DB_POOL_SIZE
                
        except Exception as e:
            # 오류 발생 시 추적 중인 세션 수 사용
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"DB pool stats error: {e}")
            checked_out = self._active_sessions
        
        # 디버깅 정보 로그
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"DB pool stats - Active sessions: {self._active_sessions}, "
                    f"Total created: {self._total_sessions_created}, "
                    f"Pool checked out: {checked_out}, Pool size: {pool_size}")
        
        return {
            'db_connections': int(checked_out),
            'db_connections_max': int(pool_size),
        }
