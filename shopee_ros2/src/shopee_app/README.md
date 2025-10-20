# Shopee App

Shopee 프론트엔드 애플리케이션 소스 코드

# 실행하기

## ROS2 환경 (권장)
1. 워크스페이스 루트에서 빌드
    ```
    colcon build --packages-select shopee_app
    ```
2. 환경 설정
    ```
    source install/setup.bash
    ```
3. 애플리케이션 실행
    ```
    ros2 run shopee_app shopee_app_gui
    ```

## 독립 실행 (개발용)
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
    python -m shopee_app.launcher
    ```

# 개발모드 실행하기 (ui 파일 자동 변환, Qt Designer 변경 시 자동 재시작)
1. 위 독립 실행을 위한 가상환경/의존성 설치 완료
2. 가상환경 활성화
    ```
    source .venv_gui/bin/activate
    ```
3. 개발 서버 실행
    ```
    python dev.py
    ```
