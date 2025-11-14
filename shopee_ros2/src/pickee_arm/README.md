# Pickee Arm

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/index.html)

`pickee_arm` 패키지는 Shopee ROS2 프로젝트의 일부로, `Pickee` 로봇의 팔 제어를 담당합니다. 이 패키지는 선반의 상품을 집어 장바구니의 지정된 위치에 놓는 기능에 중점을 둡니다. `pickee_vision` 패키지와 긴밀하게 협력하여 시각 서보(visual servoing)를 통해 정밀한 피킹 및 플레이싱 작업을 수행합니다.

## 목차

-   [노드](#노드)
-   [워크플로우](#워크플로우)
-   [통신 다이어그램](#통신-다이어그램)
-   [서비스](#서비스)
-   [구독하는 토픽](#구독하는-토픽)
-   [게시하는 토픽](#게시하는-토픽)
-   [의존성](#의존성)
-   [실행 방법](#실행-방법)
-   [기여](#기여)

## 노드

### `pickee_arm_node_for_basket`

`pickee_arm`의 메인 ROS2 노드입니다. 팔 제어, 상태 보고, `pickee_vision`과의 통신에 필요한 모든 서비스와 토픽을 관리합니다.

## 워크플로우

`pickee_arm`은 `pickee_vision` 및 `main_service` 노드와 다음과 같은 순서로 상호작용하며 상품을 피킹합니다.

1.  **피킹 시작**: `vision` 노드가 `/pickee/arm/move_start` 서비스를 호출하여 팔을 초기 피킹 준비 자세(`CHECK_SHELF_POSE`)로 이동시킵니다.
2.  **상품 위치로 이동**: `main` 노드가 `/pickee/arm/check_product` 서비스를 호출하고, 상품이 위치한 `bbox_number` (1, 2, 또는 3)를 전달합니다. 팔은 해당 그리드의 상단 뷰(`TOP_VIEW_POSE_GRID_...`)로 이동합니다. 이동이 완료되면 `/pickee/arm/is_moving` 토픽을 통해 `vision` 노드에 알립니다.
3.  **정밀 제어 (시각 서보)**: `vision` 노드는 `/pickee/arm/move_servo` 토픽을 통해 팔의 6D 좌표 값을 지속적으로 발행하여 상품을 정밀하게 조준합니다.
4.  **상품 잡기**: `vision` 노드가 `/pickee/arm/grep_product` 서비스를 호출합니다. 팔은 Z축으로 하강하여 그리퍼를 닫아 상품을 잡습니다.
5.  **장바구니에 놓기**: `main` 노드가 `/pickee/arm/place_product` 서비스를 호출합니다. 팔은 내부 카운터를 사용하여 3개의 지정된 장바구니 위치 중 하나에 순차적으로 상품을 내려놓습니다.

## 통신 다이어그램

```
[pickee_vision]                                [pickee_arm]                               [main_service]
      |                                              |                                          |
      | -- (1) /move_start (Trigger) ---------------> |                                          |
      |                                              | (Moves to CHECK_SHELF_POSE)              |
      |                                              |                                          |
      |                                              | <--- (2) /check_product (bbox_num) ------ |
      |                                              | (Moves to TOP_VIEW_POSE_GRID_...)        |
      |                                              |                                          |
      | <------------ (3) /is_moving (Bool:True) ---- |                                          |
      |                                              |                                          |
      | -- (4) /move_servo (Pose6D) ----------------> | (Visual Servoing)                        |
      |                                              |                                          |
      | -- (5) /grep_product (Trigger) -------------> | (Grasps product)                         |
      |                                              |                                          |
      |                                              | <--- (6) /place_product ----------------- |
      |                                              | (Places in basket)                       |
      |                                              |                                          |
```

## 서비스

-   **`/pickee/arm/move_start` (`std_srvs/srv/Trigger`)**
    `vision` 노드에서 호출하여 팔을 피킹 시작 자세(`STANDBY_POSE` -> `CHECK_SHELF_POSE`)로 이동시킵니다.

-   **`/pickee/arm/check_product` (`shopee_interfaces/srv/ArmCheckBbox`)**
    `main` 노드에서 호출하며, `bbox_number`에 해당하는 그리드 위로 팔을 이동시킵니다.
    -   `bbox_number` (int): `pickee_vision`이 감지한 상품의 위치를 나타내는 식별자 (1, 2, 또는 3).

-   **`/pickee/arm/grep_product` (`std_srvs/srv/Trigger`)**
    `vision` 노드에서 호출하여 현재 위치에서 Z축으로 하강하고 그리퍼를 닫아 상품을 잡도록 명령합니다.

-   **`/pickee/arm/place_product` (`shopee_interfaces/srv/ArmPlaceProduct`)**
    `main` 노드에서 호출하며, 잡고 있는 상품을 장바구니의 다음 위치에 놓습니다.

-   **`/pickee/arm/move_to_pose` (`shopee_interfaces/srv/ArmMoveToPose`)**
    지정된 정적 자세로 팔을 움직입니다. `pose_type`에 사용할 수 있는 값은 다음과 같습니다.
    -   `standby`: 대기 자세 (`STANDBY_POSE`)
    -   `lying_down`: 정지 또는 종료 시 사용되는 자세 (`LYING_DOWN_POSE`)
    -   `shelf_view`: 선반을 바라보는 자세 (`CHECK_SHELF_POSE`)

## 구독하는 토픽

-   **`/pickee/arm/move_servo` (`shopee_interfaces/msg/Pose6D`)**
    시각 서보 중 `vision` 노드로부터 팔이 이동해야 할 목표 6D 좌표를 수신합니다.
    -   `x, y, z, rx, ry, rz` (float): 팔 끝(end-effector)의 목표 좌표.

## 게시하는 토픽

-   **`/pickee/arm/is_moving` (`std_msgs/msg/Bool`)**
    `check_product` 서비스 처리 후, 팔의 이동이 완료되었음을 `vision` 노드에 알리는 신호로 사용됩니다. (`True`: 이동 완료, `False`: 이동 중 또는 대기)

-   **`/pickee/arm/real_pose` (`shopee_interfaces/msg/Pose6D`)**
    팔 끝(end-effector)의 현재 6D 좌표를 실시간으로 게시합니다.

-   **`/pickee/arm/pose_status` (`shopee_interfaces/msg/ArmPoseStatus`)**
    `move_to_pose` 서비스 처리 상태와 팔의 현재 자세 상태를 게시합니다.

-   **`/pickee/arm/pick_status` & `/pickee/arm/place_status` (`shopee_interfaces/msg/ArmTaskStatus`)**
    피킹 및 플레이싱 작업의 현재 상태(예: `in_progress`, `completed`, `failed`)를 게시합니다.

## 의존성

-   `pickee_main`
-   `pickee_vision`
-   `shopee_interfaces`

## 실행 방법

`pickee_arm` 노드는 일반적으로 `pickee_main`의 일부로 실행됩니다. 단독으로 실행하려면 다음 명령어를 사용할 수 있습니다.

```bash
ros2 run pickee_arm pickee_arm_node_for_basket
```

## 기여

이 패키지에 기여하려면 `flake8` 및 `pep257` 코드 스타일 가이드를 준수해야 합니다. 기여하기 전에 다음 테스트를 통과해야 합니다.

```bash
colcon test --packages-select pickee_arm
```
