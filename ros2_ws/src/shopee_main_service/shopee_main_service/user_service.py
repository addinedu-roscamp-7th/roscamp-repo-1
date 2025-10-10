"""
사용자 인증 및 정보 관리 서비스

사용자 로그인, 정보 조회 등의 기능을 제공합니다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .database_manager import DatabaseManager


class UserService:
    """
    사용자 서비스
    
    - 로그인 인증 (user_id + 비밀번호 검증)
    - 사용자 정보 조회
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db

    async def login(self, user_id: str, password: str) -> bool:
        """
        사용자 로그인 검증
        
        Args:
            user_id: 로그인 ID (customer.id 컬럼, 문자열)
            password: 비밀번호 (평문, TODO: 실제로는 해시 비교)
            
        Returns:
            bool: 로그인 성공 여부
            
        구현 예정:
            1. DB에서 user_id(customer.id)로 customer 조회
            2. 비밀번호 해시 비교 (bcrypt)
            3. 성공 시 customer_id(int) 반환용 세션 생성
        """
        # TODO: 실제 DB 조회 및 해시 비교
        # with self._db.session_scope() as session:
        #     customer = session.query(Customer).filter_by(id=user_id).first()
        #     if customer and verify_password(password, customer.password):
        #         return True
        return False

    async def get_user_info(self, user_id: str) -> Optional[dict]:
        """
        사용자 정보 조회
        
        Args:
            user_id: 로그인 ID (customer.id)
            
        Returns:
            dict: 사용자 정보 (이름, 나이, 주소, 알레르기 정보 등)
                  또는 None (사용자 없음)
                  
        구현 예정:
            - customer 테이블에서 user_id로 조회
            - allergy_info JOIN하여 알레르기 정보 포함
            - 응답 형식: App_vs_Main.md의 user_login_response 참고
        """
        # TODO: DB 조회 및 변환
        return None
