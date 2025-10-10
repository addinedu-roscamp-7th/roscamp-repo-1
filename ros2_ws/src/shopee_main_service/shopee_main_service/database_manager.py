"""
데이터베이스 관리자

ORM(SQLAlchemy) 세션 라이프사이클을 관리합니다.
- 세션 생성/커밋/롤백/종료 자동화
- 트랜잭션 관리
"""
from __future__ import annotations

import contextlib
from typing import Iterator


class DatabaseSession:
    """
    ORM 세션 래퍼 (Placeholder)
    
    실제로는 SQLAlchemy Session을 사용합니다.
    """

    def commit(self) -> None:  # pragma: no cover - placeholder
        """변경사항을 DB에 커밋"""
        pass

    def rollback(self) -> None:  # pragma: no cover - placeholder
        """변경사항을 롤백"""
        pass

    def close(self) -> None:  # pragma: no cover - placeholder
        """세션 종료"""
        pass


class DatabaseManager:
    """
    데이터베이스 세션 관리자
    
    SQLAlchemy ORM 세션의 생명주기를 관리합니다.
    - 세션 팩토리 제공
    - 자동 커밋/롤백
    - 리소스 정리
    """

    def __init__(self) -> None:
        # SQLAlchemy 엔진 (나중에 설정)
        self._engine = None

    def configure(self, engine: object) -> None:
        """
        DB 엔진 설정
        
        Args:
            engine: SQLAlchemy 엔진 객체
            
        사용 예:
            from sqlalchemy import create_engine
            engine = create_engine('mysql://user:pass@localhost/shopee')
            db_manager.configure(engine)
        """
        self._engine = engine

    @contextlib.contextmanager
    def session_scope(self) -> Iterator[DatabaseSession]:
        """
        세션 컨텍스트 매니저
        
        with 문과 함께 사용하여 자동으로 세션을 관리합니다.
        - 정상 종료: commit
        - 예외 발생: rollback
        - 항상: close
        
        사용 예:
            with db_manager.session_scope() as session:
                customer = session.query(Customer).filter_by(id=user_id).first()
                # 예외 없으면 자동 커밋
                # 예외 발생 시 자동 롤백
        
        Yields:
            DatabaseSession: DB 세션 객체
        """
        session = DatabaseSession()
        try:
            yield session
            session.commit()  # 성공 시 커밋
        except Exception:
            session.rollback()  # 실패 시 롤백
            raise
        finally:
            session.close()  # 항상 종료
