import rclpy
from rclpy.node import Node
from .state_machine import StateMachine
from .states.initializing import InitializingState

# 구독자(Subscriber)용 메시지 타입 임포트
from shopee_interfaces.msg import (
    PickeeMobileArrival,
    PickeeMobilePose,
    PickeeArmTaskStatus,
    PickeeVisionDetection,
    PickeeVisionObstacles,
    PickeeVisionStaffLocation,
    PickeeVisionStaffRegister
)

# 발행자(Publisher)용 메시지 타입 임포트 (Main Service 연동)
from shopee_interfaces.msg import (
    PickeeRobotStatus,
    PickeeArrival,
    PickeeProductDetection,
    PickeeCartHandover,
    PickeeProductSelection
)

# 서비스 클라이언트용 서비스 타입 임포트
from shopee_interfaces.srv import (
    PickeeMobileMoveToLocation,
    PickeeArmMoveToPose,
    PickeeArmPickProduct,
    PickeeArmPlaceProduct,
    PickeeVisionDetectProducts,
    PickeeVisionSetMode,
    PickeeVisionTrackStaff
)

# 서비스 서버용 서비스 타입 임포트 (Main Service 연동)
from shopee_interfaces.srv import (
    PickeeWorkflowStartTask,
    PickeeWorkflowMoveToSection,
    PickeeProductDetect,
    PickeeProductProcessSelection,
    PickeeWorkflowEndShopping,
    PickeeWorkflowMoveToPackaging,
    PickeeWorkflowReturnToBase,
    PickeeMainVideoStreamStart,
    PickeeMainVideoStreamStop,
    MainGetProductLocation,
    MainGetLocationPose
)


class PickeeMainController(Node):
    # Pickee 로봇의 메인 컨트롤러 노드
    
    def __init__(self):
        super().__init__('pickee_main_controller')
        
        # ROS2 파라미터 선언
        self.declare_parameter('robot_id', 1)
        self.declare_parameter('battery_threshold_available', 30.0)
        self.declare_parameter('battery_threshold_unavailable', 10.0)
        self.declare_parameter('default_linear_speed', 1.0)
        self.declare_parameter('default_angular_speed', 0.5)
        self.declare_parameter('main_service_timeout', 5.0)
        self.declare_parameter('component_service_timeout', 3.0)
        
        # 파라미터 값 가져오기
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        self.battery_threshold_available = self.get_parameter('battery_threshold_available').get_parameter_value().double_value
        self.battery_threshold_unavailable = self.get_parameter('battery_threshold_unavailable').get_parameter_value().double_value
        
        self.get_logger().info(f'Pickee Main Controller initialized with robot_id: {self.robot_id}')
        
        # 상태 기계 초기화 (초기 상태: INITIALIZING)
        initial_state = InitializingState(self)
        self.state_machine = StateMachine(initial_state)
        
        # 2.1단계: Subscriber 구현 - 내부 컴포넌트에서 발행하는 토픽 구독
        self.setup_internal_subscribers()
        
        # 2.2단계: Service Client 구현 - 내부 컴포넌트에 명령을 내리기 위한 클라이언트
        self.setup_internal_service_clients()
        
        # 3.1단계: Publisher 구현 - Main Service에 상태를 보고하기 위한 퍼블리셔
        self.setup_external_publishers()
        
        # 3.2단계: Service Server 구현 - Main Service로부터 명령을 수신하기 위한 서버
        self.setup_external_service_servers()
        
        # 로봇 상태 정보를 저장할 변수들
        self.current_battery_level = 100.0
        self.current_position_x = 0.0
        self.current_position_y = 0.0
        self.current_orientation_z = 0.0
        self.current_order_id = 0
        
        # 주기적으로 상태 기계 실행하기 위한 타이머 (10Hz)
        self.timer = self.create_timer(0.1, self.state_machine_callback)
        
        # 주기적으로 로봇 상태를 발행하기 위한 타이머 (1Hz)
        self.status_timer = self.create_timer(1.0, self.publish_robot_status)
        
        self.get_logger().info('Pickee Main Controller started')
    
    def setup_internal_subscribers(self):
        # 내부 컴포넌트 토픽 구독자 설정
        # Mobile에서 발행하는 토픽들
        self.mobile_arrival_sub = self.create_subscription(
            PickeeMobileArrival,
            '/pickee/mobile/arrival',
            self.mobile_arrival_callback,
            10
        )
        
        self.mobile_pose_sub = self.create_subscription(
            PickeeMobilePose,
            '/pickee/mobile/pose',
            self.mobile_pose_callback,
            10
        )
        
        # Arm에서 발행하는 토픽들
        self.arm_pick_status_sub = self.create_subscription(
            PickeeArmTaskStatus,
            '/pickee/arm/pick_status',
            self.arm_pick_status_callback,
            10
        )
        
        self.arm_place_status_sub = self.create_subscription(
            PickeeArmTaskStatus,
            '/pickee/arm/place_status',
            self.arm_place_status_callback,
            10
        )
        
        # Vision에서 발행하는 토픽들
        self.vision_detection_sub = self.create_subscription(
            PickeeVisionDetection,
            '/pickee/vision/detection_result',
            self.vision_detection_callback,
            10
        )
        
        self.vision_obstacles_sub = self.create_subscription(
            PickeeVisionObstacles,
            '/pickee/vision/obstacle_detected',
            self.vision_obstacles_callback,
            10
        )
        
        self.vision_staff_location_sub = self.create_subscription(
            PickeeVisionStaffLocation,
            '/pickee/vision/staff_location',
            self.vision_staff_location_callback,
            10
        )
        
        self.vision_staff_register_sub = self.create_subscription(
            PickeeVisionStaffRegister,
            '/pickee/vision/register_staff_result',
            self.vision_staff_register_callback,
            10
        )

    def setup_internal_service_clients(self):
        # 내부 컴포넌트 서비스 클라이언트 설정
        # Mobile 서비스 클라이언트
        self.mobile_move_client = self.create_client(
            PickeeMobileMoveToLocation,
            '/pickee/mobile/move_to_location'
        )
        
        # Arm 서비스 클라이언트
        self.arm_move_to_pose_client = self.create_client(
            PickeeArmMoveToPose,
            '/pickee/arm/move_to_pose'
        )
        
        self.arm_pick_product_client = self.create_client(
            PickeeArmPickProduct,
            '/pickee/arm/pick_product'
        )
        
        self.arm_place_product_client = self.create_client(
            PickeeArmPlaceProduct,
            '/pickee/arm/place_product'
        )
        
        # Vision 서비스 클라이언트
        self.vision_detect_products_client = self.create_client(
            PickeeVisionDetectProducts,
            '/pickee/vision/detect_products'
        )
        
        self.vision_set_mode_client = self.create_client(
            PickeeVisionSetMode,
            '/pickee/vision/set_mode'
        )
        
        self.vision_track_staff_client = self.create_client(
            PickeeVisionTrackStaff,
            '/pickee/vision/track_staff'
        )

    def setup_external_publishers(self):
        # Main Service에 보고하기 위한 Publisher 설정
        # 로봇 상태를 주기적으로 발행
        self.robot_status_pub = self.create_publisher(
            PickeeRobotStatus,
            '/pickee/robot_status',
            10
        )
        
        # 목적지 도착 알림
        self.arrival_notice_pub = self.create_publisher(
            PickeeArrival,
            '/pickee/arrival_notice',
            10
        )
        
        # 상품 인식 완료 알림
        self.product_detected_pub = self.create_publisher(
            PickeeProductDetection,
            '/pickee/product_detected',
            10
        )
        
        # 장바구니 교체 완료 알림
        self.cart_handover_pub = self.create_publisher(
            PickeeCartHandover,
            '/pickee/cart_handover_complete',
            10
        )
        
        # 상품 담기 완료 보고
        self.product_selection_pub = self.create_publisher(
            PickeeProductSelection,
            '/pickee/product/selection_result',
            10
        )

    def setup_external_service_servers(self):
        # Main Service로부터 명령을 수신하기 위한 Service Server 설정
        # 작업 시작 명령
        self.start_task_service = self.create_service(
            PickeeWorkflowStartTask,
            '/pickee/workflow/start_task',
            self.start_task_callback
        )
        
        # 섹션 이동 명령
        self.move_to_section_service = self.create_service(
            PickeeWorkflowMoveToSection,
            '/pickee/workflow/move_to_section',
            self.move_to_section_callback
        )
        
        # 상품 인식 명령
        self.product_detect_service = self.create_service(
            PickeeProductDetect,
            '/pickee/product/detect',
            self.product_detect_callback
        )
        
        # 상품 담기 명령
        self.process_selection_service = self.create_service(
            PickeeProductProcessSelection,
            '/pickee/product/process_selection',
            self.process_selection_callback
        )
        
        # 쇼핑 종료 명령
        self.end_shopping_service = self.create_service(
            PickeeWorkflowEndShopping,
            '/pickee/workflow/end_shopping',
            self.end_shopping_callback
        )
        
        # 포장대 이동 명령
        self.move_to_packaging_service = self.create_service(
            PickeeWorkflowMoveToPackaging,
            '/pickee/workflow/move_to_packaging',
            self.move_to_packaging_callback
        )
        
        # 복귀 명령
        self.return_to_base_service = self.create_service(
            PickeeWorkflowReturnToBase,
            '/pickee/workflow/return_to_base',
            self.return_to_base_callback
        )
        
        # 영상 송출 시작 명령
        self.video_start_service = self.create_service(
            PickeeMainVideoStreamStart,
            '/pickee/video_stream/start',
            self.video_start_callback
        )
        
        # 영상 송출 중지 명령
        self.video_stop_service = self.create_service(
            PickeeMainVideoStreamStop,
            '/pickee/video_stream/stop',
            self.video_stop_callback
        )
        
        # Main Service에 상품 위치 조회를 위한 클라이언트
        self.get_product_location_client = self.create_client(
            MainGetProductLocation,
            '/main/get_product_location'
        )

        self.get_location_pose_client = self.create_client(
            MainGetLocationPose,
            '/main/get_location_pose'
        )

    # Mobile 관련 콜백 함수들
    def mobile_arrival_callback(self, msg):
        # Mobile 도착 알림 콜백
        self.get_logger().info(f'Mobile arrival: robot_id={msg.robot_id}, location_id={msg.location_id}')
        # 상태 기계에 도착 이벤트 전달
        self.arrival_received = True
        self.arrived_location_id = msg.location_id

    def mobile_pose_callback(self, msg):
        # Mobile 위치 업데이트 콜백
        # 로봇 상태에 현재 위치 정보 업데이트
        self.current_position_x = msg.current_pose.x
        self.current_position_y = msg.current_pose.y
        self.current_orientation_z = msg.current_pose.theta
        self.current_battery_level = msg.battery_level

    # Arm 관련 콜백 함수들
    def arm_pick_status_callback(self, msg):
        # Arm 픽업 상태 콜백
        self.get_logger().info(f'Arm pick status: robot_id={msg.robot_id}, status={msg.status}')
        if msg.status == 'completed':
            # 상태 기계에 픽업 상태 이벤트 전달
            self.arm_pick_completed = True

    def arm_place_status_callback(self, msg):
        # Arm 놓기 상태 콜백
        self.get_logger().info(f'Arm place status: robot_id={msg.robot_id}, status={msg.status}')
        if msg.status == 'completed':
            # 상태 기계에 놓기 완료 이벤트 전달
            self.arm_place_completed = True

    # Vision 관련 콜백 함수들
    def vision_detection_callback(self, msg):
        # Vision 상품 인식 결과 콜백
        self.get_logger().info(f'Vision detection: robot_id={msg.robot_id}, success={msg.success}')
        # 상태 기계에 인식 결과 이벤트 전달
        self.detection_result = msg

    def vision_obstacles_callback(self, msg):
        # Vision 장애물 감지 콜백
        self.get_logger().info(f'Vision obstacles: robot_id={msg.robot_id}, count={len(msg.obstacles)}')
        # TODO: Mobile 제어에 장애물 정보 반영

    def vision_staff_location_callback(self, msg):
        # Vision 직원 위치 콜백
        # TODO: 직원 추종 로직에 위치 정보 사용
        pass

    def vision_staff_register_callback(self, msg):
        # Vision 직원 등록 결과 콜백
        self.get_logger().info(f'Vision staff register: robot_id={msg.robot_id}, success={msg.success}')
        if msg.success:
            # TODO: 상태 기계에 직원 등록 완료 이벤트 전달
            pass

    # Service Client 래퍼 함수들
    async def call_mobile_move_to_location(self, location_id, target_pose, global_path=None, navigation_mode='normal'):
        # Mobile에 위치 이동 명령
        request = PickeeMobileMoveToLocation.Request()
        request.robot_id = self.robot_id
        request.order_id = 0  # TODO: 현재 주문 ID로 설정
        request.location_id = location_id
        request.target_pose = target_pose
        request.global_path = global_path or []
        request.navigation_mode = navigation_mode
        
        if not self.mobile_move_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Mobile move service not available')
            return False
        
        try:
            future = self.mobile_move_client.call_async(request)
            response = await future
            return response.success
        except Exception as e:
            self.get_logger().error(f'Mobile move service call failed: {str(e)}')
            return False

    async def call_arm_move_to_pose(self, pose_type):
        # Arm에 자세 변경 명령
        request = PickeeArmMoveToPose.Request()
        request.robot_id = self.robot_id
        request.order_id = 0  # TODO: 현재 주문 ID로 설정
        request.pose_type = pose_type
        
        if not self.arm_move_to_pose_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Arm move to pose service not available')
            return False
        
        try:
            future = self.arm_move_to_pose_client.call_async(request)
            response = await future
            return response.success
        except Exception as e:
            self.get_logger().error(f'Arm move to pose service call failed: {str(e)}')
            return False

    async def call_arm_pick_product(self, product_id, target_position):
        # Arm에 상품 픽업 명령
        request = PickeeArmPickProduct.Request()
        request.robot_id = self.robot_id
        request.order_id = 0  # TODO: 현재 주문 ID로 설정
        request.product_id = product_id
        request.target_position = target_position
        
        if not self.arm_pick_product_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Arm pick product service not available')
            return False
        
        try:
            future = self.arm_pick_product_client.call_async(request)
            response = await future
            return response.accepted
        except Exception as e:
            self.get_logger().error(f'Arm pick product service call failed: {str(e)}')
            return False

    async def call_arm_place_product(self, product_id):
        # Arm에 상품 놓기 명령
        request = PickeeArmPlaceProduct.Request()
        request.robot_id = self.robot_id
        request.order_id = 0  # TODO: 현재 주문 ID로 설정
        request.product_id = product_id
        
        if not self.arm_place_product_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Arm place product service not available')
            return False
        
        try:
            future = self.arm_place_product_client.call_async(request)
            response = await future
            return response.accepted
        except Exception as e:
            self.get_logger().error(f'Arm place product service call failed: {str(e)}')
            return False

    async def call_vision_detect_products(self, product_ids):
        # Vision에 상품 인식 명령
        request = PickeeVisionDetectProducts.Request()
        request.robot_id = self.robot_id
        request.order_id = 0  # TODO: 현재 주문 ID로 설정
        request.product_ids = product_ids
        
        if not self.vision_detect_products_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Vision detect products service not available')
            return False
        
        try:
            future = self.vision_detect_products_client.call_async(request)
            response = await future
            return response.success
        except Exception as e:
            self.get_logger().error(f'Vision detect products service call failed: {str(e)}')
            return False

    async def call_vision_set_mode(self, mode):
        # Vision 모드 설정 명령
        request = PickeeVisionSetMode.Request()
        request.robot_id = self.robot_id
        request.mode = mode
        
        if not self.vision_set_mode_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Vision set mode service not available')
            return False
        
        try:
            future = self.vision_set_mode_client.call_async(request)
            response = await future
            return response.success
        except Exception as e:
            self.get_logger().error(f'Vision set mode service call failed: {str(e)}')
            return False

    async def call_vision_track_staff(self, track):
        # Vision 직원 추종 제어 명령
        request = PickeeVisionTrackStaff.Request()
        request.robot_id = self.robot_id
        request.track = track
        
        if not self.vision_track_staff_client.wait_for_service(timeout_sec=self.get_parameter('component_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Vision track staff service not available')
            return False
        
        try:
            future = self.vision_track_staff_client.call_async(request)
            response = await future
            return response.success
        except Exception as e:
            self.get_logger().error(f'Vision track staff service call failed: {str(e)}')
            return False

    # Publisher 메소드들
    def publish_robot_status(self):
        # 로봇 상태를 주기적으로 발행
        msg = PickeeRobotStatus()
        msg.robot_id = self.robot_id
        msg.state = self.state_machine.get_current_state_name()
        msg.battery_level = self.current_battery_level
        msg.current_order_id = self.current_order_id
        msg.position_x = self.current_position_x
        msg.position_y = self.current_position_y
        msg.orientation_z = self.current_orientation_z
        
        self.robot_status_pub.publish(msg)

    def publish_arrival_notice(self, location_id, section_id=0):
        # 목적지 도착 알림 발행
        msg = PickeeArrival()
        msg.robot_id = self.robot_id
        msg.order_id = self.current_order_id
        msg.location_id = location_id
        msg.section_id = section_id
        
        self.arrival_notice_pub.publish(msg)
        self.get_logger().info(f'Published arrival notice: location_id={location_id}')

    def publish_product_detected(self, products):
        # 상품 인식 완료 알림 발행
        msg = PickeeProductDetection()
        msg.robot_id = self.robot_id
        msg.order_id = self.current_order_id
        msg.products = products
        
        self.product_detected_pub.publish(msg)
        self.get_logger().info(f'Published product detection: {len(products)} products')

    def publish_cart_handover_complete(self):
        # 장바구니 교체 완료 알림 발행
        msg = PickeeCartHandover()
        msg.robot_id = self.robot_id
        msg.order_id = self.current_order_id
        
        self.cart_handover_pub.publish(msg)
        self.get_logger().info('Published cart handover complete')

    def publish_product_selection_result(self, product_id, success, quantity, message=''):
        # 상품 담기 완료 보고 발행
        msg = PickeeProductSelection()
        msg.robot_id = self.robot_id
        msg.order_id = self.current_order_id
        msg.product_id = product_id
        msg.success = success
        msg.quantity = quantity
        msg.message = message
        
        self.product_selection_pub.publish(msg)
        self.get_logger().info(f'Published product selection result: product_id={product_id}, success={success}')

    async def get_product_location(self, product_id):
        # Main Service에서 상품 위치 조회
        request = MainGetProductLocation.Request()
        request.product_id = product_id
        
        if not self.get_product_location_client.wait_for_service(timeout_sec=self.get_parameter('main_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Get product location service not available')
            return None
        
        try:
            future = self.get_product_location_client.call_async(request)
            response = await future
            if response.success:
                return {
                    'warehouse_id': response.warehouse_id,
                    'section_id': response.section_id
                }
            return None
        except Exception as e:
            self.get_logger().error(f'Get product location service call failed: {str(e)}')
            return None

    async def call_get_location_pose(self, location_id):
        # Main Service에서 위치 Pose 조회
        request = MainGetLocationPose.Request()
        request.location_id = location_id
        
        if not self.get_location_pose_client.wait_for_service(timeout_sec=self.get_parameter('main_service_timeout').get_parameter_value().double_value):
            self.get_logger().error('Get location pose service not available')
            return None
        
        try:
            future = self.get_location_pose_client.call_async(request)
            response = await future
            if response.success:
                return response.pose
            return None
        except Exception as e:
            self.get_logger().error(f'Get location pose service call failed: {str(e)}')
            return None

    # Service Server 콜백 함수들
    def start_task_callback(self, request, response):
        # 작업 시작 명령 콜백
        self.get_logger().info(f'Received start task: robot_id={request.robot_id}, order_id={request.order_id}')
        
        # 현재 주문 ID 업데이트
        self.current_order_id = request.order_id
        
        # 상품 목록을 내부 변수에 저장
        self.product_list = request.product_list
        self.remaining_products = list(request.product_list)  # 복사본 생성
        
        # 첫 번째 상품의 위치로 이동 시작
        if self.remaining_products:
            first_product = self.remaining_products[0]
            self.target_location_id = first_product.location_id
            self.target_product_ids = [first_product.product_id]
            
            # MOVING_TO_SHELF 상태로 전환
            from .states.moving_to_shelf import MovingToShelfState
            new_state = MovingToShelfState(self)
            self.state_machine.transition_to(new_state)
        
        response.success = True
        response.message = 'Task started successfully'
        return response

    def move_to_section_callback(self, request, response):
        # 섹션 이동 명령 콜백
        self.get_logger().info(f'Received move to section: location_id={request.location_id}, section_id={request.section_id}')
        
        # TODO: 상태 기계에 섹션 이동 이벤트 전달
        
        response.success = True
        response.message = 'Move to section command accepted'
        return response

    def product_detect_callback(self, request, response):
        # 상품 인식 명령 콜백
        self.get_logger().info(f'Received product detect: product_ids={request.product_ids}')
        
        # TODO: Vision에 상품 인식 명령 전달
        
        response.success = True
        response.message = 'Product detection started'
        return response

    def process_selection_callback(self, request, response):
        # 상품 담기 명령 콜백
        self.get_logger().info(f'Received process selection: product_id={request.product_id}, bbox_number={request.bbox_number}')
        
        # 상태 기계에 상품 선택 이벤트 전달
        self.selection_request = request
        
        response.success = True
        response.message = 'Product selection processing started'
        return response

    def end_shopping_callback(self, request, response):
        # 쇼핑 종료 명령 콜백
        self.get_logger().info(f'Received end shopping: robot_id={request.robot_id}, order_id={request.order_id}')
        
        # TODO: 상태 기계에 쇼핑 종료 이벤트 전달
        
        response.success = True
        response.message = 'Shopping ended, moving to packaging'
        return response

    def move_to_packaging_callback(self, request, response):
        # 포장대 이동 명령 콜백
        self.get_logger().info(f'Received move to packaging: location_id={request.location_id}')
        
        # TODO: 상태 기계에 포장대 이동 이벤트 전달
        
        response.success = True
        response.message = 'Moving to packaging area'
        return response

    def return_to_base_callback(self, request, response):
        # 복귀 명령 콜백
        self.get_logger().info(f'Received return to base: location_id={request.location_id}')
        
        # TODO: 상태 기계에 복귀 이벤트 전달
        
        response.success = True
        response.message = 'Returning to base'
        return response

    def video_start_callback(self, request, response):
        # 영상 송출 시작 명령 콜백
        self.get_logger().info(f'Received video start: user_type={request.user_type}, user_id={request.user_id}')
        
        # TODO: Vision에 영상 송출 시작 명령 전달
        
        response.success = True
        response.message = 'Video streaming started'
        return response

    def video_stop_callback(self, request, response):
        # 영상 송출 중지 명령 콜백
        self.get_logger().info(f'Received video stop: user_type={request.user_type}, user_id={request.user_id}')
        
        # TODO: Vision에 영상 송출 중지 명령 전달
        
        response.success = True
        response.message = 'Video streaming stopped'
        return response

    def state_machine_callback(self):
        # 상태 기계를 주기적으로 실행하는 콜백 함수
        try:
            self.state_machine.execute()
        except Exception as e:
            self.get_logger().error(f'State machine execution error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PickeeMainController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
