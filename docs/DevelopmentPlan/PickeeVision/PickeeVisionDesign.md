# Pickee Vision 관여 시나리오

SC_02_2 장애물 회피
 - 기능: 장애물 인식, 장애물 분류(타입, 위치, 속도, 거리)
 - 통신_토픽: 장애물 감지 알림 (/pickee/vision/obstacle_detected)
SC_02_3 매대에서 상품 선택
 - 기능: 상품 인식, 송출 중인 영상에 bbox 그리기
 - 통신_서비스: 매대 상품 인식 (/pickee/vision/detect_products)
 - 통신_토픽: 매대 상품 인식 (/pickee/vision/dectection_result)
SC_02_4 상품 장바구니 담기
 - 기능: 해당 상품 장바구니 확인
 - 통신_서비스: 장바구니 내 특정 상품 확인 (pickee/vision/check_product_in_cart)

SC_03_2 장바구니 교체
 - 기능: 장바구니 유무 확인
 - 통신_서비스: 장바구니 존재 확인 (/pickee/vision/check_cart_presence)

SC_05_1 관리자 모니터링 로봇 시야 확인 UDP
 - 통신_서비스: 영상 송출 시작 (/pickee/vision/video_stream_start)
 - 통신_서비스: 영상 송출 종료 (/pickee/vision/video_stream_stop)
 - 통신_UDP: 영상 송출 (video_frame)

SC_06_2 인식 및 추종
 - 기능: 인식 모드 전환 (주행모드 -> 등록모드), 정면 특징 추출, 후면 특징 추출
 - 통신_서비스: 인식 모드 전환 요청 (/pickee/vision/set_mode)
 	      직원 등록 요청 (/pickee/vision/register_staff)
 	      음성 송출 요청 (/pickee/tts_request, text="카메라 정면을 봐주세요") - 정면 특징 추출
 	      음성 송출 요청 (/pickee/tts_request, text="뒤로 돌아주세요") - 후면 특징 추출
 - 통신_토픽: 등록 결과 전송 (/pickee/vision/register_staff_result)
 - 기능: 카메라 영상에서 직원 위치 식별
 - 통신_토픽: 직원의 상대 위치 전달 (/pickee/vision/staff_location)
 	      
SC_06_3 음성 명령
 - 기능: 추종 중지
 - 통신_서비스: 직원 추종 중지 (/pickee/vision/track_staff)

 