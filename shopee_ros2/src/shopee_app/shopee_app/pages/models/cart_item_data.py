from dataclasses import dataclass
from pathlib import Path


@dataclass
class CartItemData:
    product_id: int
    name: str
    quantity: int
    price: int
    image_path: Path
    is_selected: bool = True

    @property
    def total_price(self) -> int:
        return self.price * self.quantity
