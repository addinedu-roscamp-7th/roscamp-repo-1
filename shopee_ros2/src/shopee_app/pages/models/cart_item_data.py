from dataclasses import dataclass
from pathlib import Path


@dataclass
class CartItemData:
    product_id: int
    name: str
    quantity: int
    price: int
    
