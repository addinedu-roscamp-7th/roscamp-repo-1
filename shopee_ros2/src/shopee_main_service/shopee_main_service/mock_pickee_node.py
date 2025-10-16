"""
Pickee 전용 Mock 노드 실행 스크립트
"""
from __future__ import annotations

from .mock_robot_node import run_mock_robot


def main() -> None:
    """Pickee 전용 Mock 노드를 실행한다."""
    run_mock_robot(default_mode='pickee')


if __name__ == '__main__':
    main()
