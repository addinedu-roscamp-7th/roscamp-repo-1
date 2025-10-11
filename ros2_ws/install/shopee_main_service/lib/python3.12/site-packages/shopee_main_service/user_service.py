"""
사용자 인증 및 정보 관리 서비스

사용자 로그인, 정보 조회 등의 기능을 제공합니다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from passlib.context import CryptContext

from .database_models import Customer

if TYPE_CHECKING:
    from .database_manager import DatabaseManager

# 비밀번호 해싱을 위한 컨텍스트 설정
# bcrypt 알고리즘을 사용하며, deprecated="auto"는 보안 강화를 위해 필요시 자동 업그레이드를 지원합니다.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
            password: 비밀번호 (평문)
            
        Returns:
            bool: 로그인 성공 여부
        """
        with self._db.session_scope() as session:
            # 데이터베이스에서 사용자 ID로 고객 정보 조회
            customer = session.query(Customer).filter_by(id=user_id).first()
            
            # 고객 정보가 존재하고, 입력된 비밀번호가 저장된 해시와 일치하는지 확인
            if customer and pwd_context.verify(password, customer.password):
                # TODO: 로그인 성공 시, 사용자 세션 또는 토큰 생성 로직 추가
                return True
        
        return False

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호와 해시된 비밀번호를 비교합니다."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """비밀번호를 해시 처리합니다."""
        return pwd_context.hash(password)

    async def get_user_info(self, user_id: str) -> Optional[dict]:
        """
        사용자 정보 조회
        
        Args:
            user_id: 로그인 ID (customer.id)
            
        Returns:
            dict: App_vs_Main.md의 user_login_response 형식에 맞는 사용자 정보
        """
        with self._db.session_scope() as session:
            customer = (
                session.query(Customer)
                .join(AllergyInfo, Customer.allergy_info_id == AllergyInfo.allergy_info_id)
                .filter(Customer.id == user_id)
                .first()
            )
            
            if not customer:
                return None
            
            # 응답 형식에 맞게 데이터 구성
            return {
                "user_id": customer.id,
                "name": customer.name,
                "gender": customer.gender,
                "age": customer.age,
                "address": customer.address,
                "allergy_info": {
                    "nuts": customer.allergy_info.nuts,
                    "milk": customer.allergy_info.milk,
                    "seafood": customer.allergy_info.seafood,
                    "soy": customer.allergy_info.soy,
                    "peach": customer.allergy_info.peach,
                    "gluten": customer.allergy_info.gluten,
                    "eggs": customer.allergy_info.eggs,
                },
                "is_vegan": customer.is_vegan,
            }
