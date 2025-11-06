from dataclasses import dataclass
from pathlib import Path


@dataclass
class CartItemData:
    '''사용자 장바구니 항목을 나타내는 모델.'''

    product_id: int
    name: str
    quantity: int
    price: int
    image_path: Path
    is_selected: bool = True

    @property
    def total_price(self) -> int:
        '''수량과 단가를 곱해 합계를 계산한다.'''
        return self.price * self.quantity
