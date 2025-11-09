from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from shopee_app.pages.models.allergy_info_data import AllergyInfoData


@dataclass(frozen=True)
class ProductData:
    '''상품 기본 정보와 알레르기 세부사항을 담는 데이터 모델.'''

    product_id: int
    name: str
    category: str
    price: int
    discount_rate: int
    allergy_info_id: int
    is_vegan_friendly: bool
    section_id: int
    warehouse_id: int
    length: int
    width: int
    height: int
    weight: int
    fragile: bool
    image_path: Path
    allergy_info: AllergyInfoData | None = field(default=None)

    @property
    def discounted_price(self) -> int:
        '''할인율을 적용한 실제 판매가를 계산한다.'''
        return int(self.price * (100 - self.discount_rate) / 100)
