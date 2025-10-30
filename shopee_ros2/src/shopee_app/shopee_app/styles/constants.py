# 색상 상수
COLORS = {
    "primary": "#FF3134",
    "primary_dark": "#cc2829",
    "gray_light": "#EEEEEE",
    "gray": "#999999",
    "gray_dark": "#666666",
}

# 폰트 스타일
FONTS = {
    "small": "10pt",
    "normal": "12pt",
    "large": "15pt",
}

# 자주 사용되는 스타일 조합
STYLES = {
    "primary_button": f"""
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    """,
    "secondary_button": f"""
        background-color: transparent;
        color: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
        border-radius: 4px;
        padding: 8px 16px;
    """,
    "info_tag": f"""
        background-color: {COLORS['gray_light']};
        color: {COLORS['gray_dark']};
        border-radius: 3px;
        padding: 2px 6px;
    """,
    "pay_button": f"""
        background-color: rgba(255, 49, 52, 0.05);
        color: {COLORS['primary']};
        border: 1px solid {COLORS['primary']};
        border-radius: 4px;
        padding: 8px 16px;
    """,
}

# 레이아웃 관련 상수
SPACING = {
    "small": 4,
    "normal": 8,
    "large": 16,
}

# 테마 설정
THEME = {
    "light": {
        "background": "white",
        "text": "#333333",
        "border": "#DDDDDD",
    },
    "dark": {
        "background": "#333333",
        "text": "white",
        "border": "#666666",
    },
}
