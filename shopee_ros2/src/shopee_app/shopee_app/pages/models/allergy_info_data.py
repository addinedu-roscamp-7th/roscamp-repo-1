from dataclasses import dataclass


@dataclass(frozen=True)
class AllergyInfoData:
    '''상품에 포함된 알레르기 유발 성분 여부를 저장한다.'''

    allergy_info_id: int
    nuts: bool
    milk: bool
    seafood: bool
    soy: bool
    peach: bool
    gluten: bool
    eggs: bool
