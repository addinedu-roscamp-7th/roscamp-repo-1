"""
사용자 인증 및 정보 관리 서비스

사용자 로그인, 정보 조회 등의 기능을 제공합니다.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

# passlib이 내부적으로 Python 3.13+에서 deprecated될 crypt 모듈을 사용하는 것에 대한 경고를 억제
warnings.filterwarnings("ignore", message="'crypt' is deprecated")

from passlib.context import CryptContext

from .database_models import Admin, AllergyInfo, Customer

if TYPE_CHECKING:
    from .database_manager import DatabaseManager

# 비밀번호 해싱을 위한 컨텍스트 설정
# bcrypt 알고리즘만 사용하도록 명시적으로 설정하여 Python 3.13+의 crypt 모듈 deprecation 경고를 방지합니다.
pwd_context = CryptContext(
    schemes=["bcrypt"],
    bcrypt__default_rounds=12,  # bcrypt의 기본 라운드 수 명시
    bcrypt__ident="2b"  # bcrypt 버전 2b 사용 (최신 버전)
)


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
        사용자 로그인 검증 (admin 또는 customer)

        Args:
            user_id: 로그인 ID (admin.id 또는 customer.id 컬럼, 문자열)
            password: 비밀번호 (평문)

        Returns:
            bool: 로그인 성공 여부
        """
        with self._db.session_scope() as session:
            # 먼저 admin 테이블에서 검색
            admin = session.query(Admin).filter_by(id=user_id).first()
            if admin:
                # admin은 비밀번호가 평문으로 저장되어 있을 수 있음 (테스트용)
                # bcrypt 해시 확인을 시도하고, 실패하면 평문 비교
                try:
                    if pwd_context.verify(password, admin.password):
                        return True
                except ValueError as e:
                    # 해시가 아닌 평문일 경우 발생
                    import logging
                    logging.getLogger(__name__).debug(f"Admin password is not hashed for user {user_id}: {e}")
                    if admin.password == password:
                        return True
                except Exception as e:
                    # 기타 예외 (손상된 해시 등)
                    import logging
                    logging.getLogger(__name__).error(f"Error verifying admin password for {user_id}: {type(e).__name__}: {e}")
                    return False

            # admin이 아니면 customer 테이블에서 검색
            customer = session.query(Customer).filter_by(id=user_id).first()
            if customer:
                # 고객 정보가 존재하고, 입력된 비밀번호가 저장된 해시와 일치하는지 확인
                try:
                    if pwd_context.verify(password, customer.password):
                        return True
                except ValueError as e:
                    # 해시가 아닌 평문일 경우 발생
                    import logging
                    logging.getLogger(__name__).debug(f"Customer password is not hashed for user {user_id}: {e}")
                    if customer.password == password:
                        return True
                except Exception as e:
                    # 기타 예외 (손상된 해시 등)
                    import logging
                    logging.getLogger(__name__).error(f"Error verifying customer password for {user_id}: {type(e).__name__}: {e}")
                    return False

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
