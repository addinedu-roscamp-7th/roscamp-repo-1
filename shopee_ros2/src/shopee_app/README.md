# Shopee App

Shopee 프론트엔드 애플리케이션 소스 코드

# 실행하기
1. 가상환경 생성
    ```
    python3 -m venv .venv_gui
    ```
2. 가상환경 활성화
    ```
    source .venv_gui/bin/activate
    ```
3. 의존성 설치
    ```
    pip install -r requirements.txt
    ```
4. 애플리케이션 실행
    ```
    python app.py
    ```


# 개발모드 실행하기 (ui 파일 자동 변환, designer 변경 저장 시 자동 재시작)
1. 위 가상환경 및 의존성 설치를 선행
2. 가상환경 활성화
    ```
    source .venv_gui/bin/activate
    ```
3. 개발 서버 실행
    ```
    python dev.py
    ```


# 초기 설정
1. python3 -m venv .venv_gui
2. source .venv_gui/bin/activate
3. pip install -r requirements.txt
4. python app.py 또는 python dev.py
