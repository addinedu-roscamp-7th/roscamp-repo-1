from typing import Dict

# 알러지 정보 한글 매핑
ALLERGY_NAME_MAP = {
    "nuts": "견과류",
    "milk": "유제품",
    "seafood": "해산물",
    "soy": "대두",
    "peach": "복숭아",
    "gluten": "글루텐",
    "eggs": "계란",
}


def get_matching_allergies(
    user_allergy: Dict[str, bool], product_allergy: Dict[str, bool]
) -> str:
    """
    사용자의 알러지 정보와 상품의 알러지 정보를 비교하여 매칭되는 알러지 항목을 반환합니다.

    Args:
        user_allergy: 사용자 알러지 정보 딕셔너리
        product_allergy: 상품 알러지 정보 딕셔너리

    Returns:
        str: 매칭된 알러지 항목들을 쉼표로 구분한 문자열 (매칭 없으면 빈 문자열)
    """
    matched = []

    for key, has_allergen in product_allergy.items():
        # 상품에 해당 알러지 유발 성분이 있고
        # 사용자가 해당 알러지가 있는 경우
        if has_allergen and user_allergy.get(key, False):
            matched.append(ALLERGY_NAME_MAP[key])

    return ", ".join(matched)


def get_vegan_status(is_vegan_friendly: bool) -> str:
    """
    비건 상태를 문자열로 반환합니다.

    Args:
        is_vegan_friendly: 비건 친화 여부

    Returns:
        str: '비건 음식' 또는 '해당 없음'
    """
    return "비건 음식" if is_vegan_friendly else "해당 없음"
