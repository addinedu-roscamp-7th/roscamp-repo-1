from dataclasses import dataclass


@dataclass(frozen=True)
class AllergyInfoData:
    allergy_info_id: int
    nuts: bool
    milk: bool
    seafood: bool
    soy: bool
    peach: bool
    gluten: bool
    eggs: bool
