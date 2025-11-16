# Pickee Vision

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/index.html)

`pickee_vision` 패키지는 `Pickee` 로봇의 시각 인지 기능을 담당합니다. 이 패키지는 두 가지 주요 기능을 수행합니다. (1) 매대의 상품을 탐지하고 장바구니의 상태를 확인하는 일반적인 객체 인식 기능과 (2) 특정 상품을 정밀하게 집기 위한 Visual Servoing 기반의 고정밀 Picking 시퀀스.

## 목차

-   [노드](#노드)
-   [핵심 기능](#핵심-기능)
-   [고정밀 피킹 워크플로우](#고정밀-피킹-워크플로우)
-   [서비스](#서비스)
-   [토픽](#토픽)
-   [사용 모델](#사용-모델)
-   [의존성](#의존성)
-   [실행 방법](#실행-방법)

## 노드

### `final_pickee_vision_node`

`pickee_vision`의 모든 기능을 통합하여 실행하는 메인 노드입니다. 두 대의 카메라로부터 영상을 입력받아 상품 탐지, 장바구니 상태 분석, Visual Servoing 등 다양한 컴퓨터 비전 작업을 수행하고 다른 노드와 통신합니다.

## 핵심 기능

### 1. 매대 및 장바구니 분석

독립적으로 호출될 수 있는 서비스들을 통해 매대의 상품 목록을 분석하거나 장바구니의 상태를 확인합니다.

-   **상품 탐지**: `YoloDetector`를 사용하여 매대 위 상품들의 위치와 종류를 탐지합니다. (`/pickee/vision/detect_products`)
-   **장바구니 내 상품 확인**: 장바구니 안에 특정 상품이 있는지 확인합니다. (`/pickee/vision/check_product_in_cart`)
-   **장바구니 존재 여부 확인**: `CnnClassifier`를 사용하여 로봇 앞에 빈 장바구니가 있는지 확인합니다. (`/pickee/vision/check_cart_presence`)

### 2. 고정밀 Picking (Visual Servoing)

`PoseCNN` 모델과 PID 제어기를 사용하여 특정 상품을 매우 정밀하게 집는 기능입니다. 이 기능은 `pickee_arm` 노드와 긴밀하게 협력하는 복잡한 상태 머신을 통해 관리됩니다.

## 고정밀 Picking 워크플로우

고정밀 Picking은 `pickee_main`, `pickee_vision`, `pickee_arm` 세 노드 간의 긴밀한 협력을 통해 이루어집니다. `pickee_main`이 전체 과정을 조율하는 역할을 합니다.

```
+--------------+      +--------------------------+      +------------------+
| pickee_main  |      |      pickee_vision       |      |    pickee_arm    |
+--------------+      +--------------------------+      +------------------+
       |                           |                           |
       | --(1) Starts sequence --> | --(2) Commands Arm -----> |
       | (e.g. /pick_product)      |   (e.g. /move_start)      |
       |                           |                           |
       |                           | <----(3) Arm is Ready --- |
       |                           |   (e.g. /is_moving)       |
       |                           |                           |
       |                           | --(4) Visual Servoing --> |
       |                           |   (/move_servo)           |
       |                           |                           |
       |                           | --(5) Grasps Product ---> |
       |                           |   (/grep_product)         |
       |                           |                           |
       | --(6) Places Product -------------------------------->|
       |   (/place_product)        |                           |
       |                           |                           |
```

1.  **시퀀스 시작**: `pickee_main` 노드가 `pickee_vision`의 `/pickee/arm/pick_product` 서비스를 호출하여 특정 상품에 대한 Picking 시퀀스를 시작합니다.
2.  **팔 초기 이동**: `pickee_vision`은 `pickee_arm`의 `/move_start` 서비스를 호출하여 팔을 매대 보기 자세로 움직이도록 명령합니다.
3.  **팔 준비 완료 신호**: `pickee_arm`은 이동이 완료되면 `/is_moving` 토픽으로 `True`를 게시하여 `pickee_vision`에 준비되었음을 알립니다.
4.  **Visual Servoing 루프**: `pickee_vision`은 `PoseCNN`을 이용한 Visual Servoing 루프를 시작합니다. 현재 이미지와 팔의 실제 좌표(`real_pose`)를 바탕으로 목표 좌표 오차를 계산하고, PID 제어기를 통해 보정된 목표 좌표를 `/move_servo` 토픽으로 계속해서 `pickee_arm`에 발행합니다.
5.  **상품 잡기**: `pickee_vision`이 목표 지점에 수렴했다고 판단하면, `/grep_product` 서비스를 호출하여 `pickee_arm`이 상품을 잡도록 합니다.
6.  **상품 놓기**: `pickee_main`이 `pickee_arm`의 `/place_product` 서비스를 호출하여 잡은 상품을 장바구니에 놓도록 명령합니다.

## 서비스

### 제공하는 서비스

-   **`/pickee/arm/pick_product` (`ArmPickProduct`)**: 고정밀 picking 시퀀스를 시작하는 메인 서비스입니다.
-   **`/pickee/vision/detect_products` (`PickeeVisionDetectProducts`)**: 매대의 상품들을 탐지하고 결과를 반환합니다.
-   **`/pickee/vision/check_product_in_cart` (`PickeeVisionCheckProductInCart`)**: 장바구니 안에 특정 상품이 있는지 확인합니다.
-   **`/pickee/vision/check_cart_presence` (`VisionCheckCartPresence`)**: 로봇 위에 빈 장바구니가 있는지 확인합니다.
-   **`/pickee/vision/video_stream_start` / `_stop`**: UDP 비디오 스트리밍을 시작/중지합니다.

### 사용하는 서비스 (Clients)

-   **`/pickee/arm/move_start` (`Trigger`)**: `pickee_arm`에게 매대 확인 자세로 이동하라고 명령합니다.
-   **`/pickee/arm/grep_product` (`Trigger`)**: `pickee_arm`에게 상품을 잡으라고 명령합니다.

## 토픽

### 게시하는 토픽

-   **`/pickee/vision/detection_result` (`PickeeVisionDetection`)**: `detect_products` 서비스의 상세 탐지 결과를 게시합니다.
-   **`/pickee/vision/cart_check_result` (`PickeeVisionCartCheck`)**: `check_product_in_cart` 서비스의 결과를 게시합니다.
-   **`/pickee/arm/move_servo` (`Pose6D`)**: Visual Servoing 중 팔이 움직여야 할 보정된 6D 좌표를 게시합니다.

### 구독하는 토픽

-   **`/pickee/arm/is_moving` (`Bool`)**: `pickee_arm`이 상품을 집기 위한 위치로 이동을 완료했는지 확인하는 신호를 받습니다.
-   **`/pickee/arm/real_pose` (`Pose6D`)**: `pickee_arm`으로부터 팔 끝의 실시간 좌표를 받습니다.

## 사용 모델

-   **`YoloDetector`**: 매대 위 또는 장바구니 안 상품들의 객체를 탐지하는 YOLO 기반 모델.
-   **`CnnClassifier`**: 카트의 존재 여부를 분류하는 CNN 모델.
-   **`PoseCNN`**: 특정 상품(예: 생선, 이클립스)의 6D Pose를 추정하여 Visual Servoing에 사용되는 CNN 모델.

## 의존성

-   `pickee_arm`
-   `shopee_interfaces`

## 실행 방법

```bash
ros2 run pickee_vision final_pickee_vision_node
```
