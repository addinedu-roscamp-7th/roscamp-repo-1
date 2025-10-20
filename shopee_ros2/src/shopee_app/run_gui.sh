#!/usr/bin/env bash
# 안전 옵션 설정(단, ROS setup 중엔 -u 해제)
set -e
set -o pipefail

# 1) ROS2 환경 (unbound 변수 허용)
set +u
source /opt/ros/jazzy/setup.bash
set -u

# 2) 가상환경
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv_gui/bin/activate"

# 3) 앱 실행
exec python -m shopee_app.launcher
