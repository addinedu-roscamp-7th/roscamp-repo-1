# LLM 응답을 받아오기 위한 GET_LLM_Response 클래스 import
from LLM_commu import GET_LLM_Response
# STT_Module 사용을 위한 STT_Module 클래스 import
from STT_module import STT_Module
# ROS 토픽 발행을 위한 ROS_Util 클래스 import
from ROS_commu import ROS_Util
# TTS 사용을 위해 TTS_Util 클래스 import
from TTS_commu import TTS_Util

# 비동기 코드 실행을 위한 asyncio 라이브러리 import
import asyncio

# 메인함수 선언
def main():
    # stt 관련 클래스 객체 선언
    stt = STT_Module()
    # llm 관련 객체 선언
    llm = GET_LLM_Response()
    # ros 관련 객체 선언
    ros = ROS_Util()
    # TTS 관련 객체 선언
    tts = TTS_Util()
    # 자율주행 목적지 배열 선언
    allowed_places = {"반납함","신선식품","기성품","과자","포장대","계산대"}

    # 무한 반복
    while True:
        # 에러가 없으면
        try:
            # 사용자 음성으로 stt를 수행
            stt_result = stt.stt_use()
            # STT 결과 print
            print("[STT]", stt_result)
            # STT결과에 종료가 있거나 대화 종료가 있으면 루프 종료
            if ("종료" in stt_result) or ("대화 종료" in stt_result):
                break
            # LLM에 STT결과를 입력하고 답변을 llm_result에 저장
            llm_result = llm.llm_answer(stt_result)
            # llm 결과 print
            print("[LLM]",llm_result)
            # 만약 llm 답변이 문자열이라면
            if isinstance(llm_result, str):
                # 답변만 추출
                result = llm_result.strip()
                # 답변에 자율주행 목적지 배열안에 있으면
                if result in allowed_places:
                    # 장소 이동 명령 수신 로그 출력
                    print("[LLM] 장소 이동 명령 수신")
                    text = (f"네, {result}로 이동하겠습니다.")
                    # TTS로 장소 이동 응답
                    asyncio.run(tts.speak(text))
                    # llm_result 결과를 String 형식으로 publish 
                    # topic명 : /move_place
                    ros.publish_string(llm_result)
                # LLM답변이 following이라면
                # lower : 소문자로 변환
                elif result.lower() == "following":
                    # 팔로잉 명령 수신 로그 출력
                    print("[LLM] 팔로잉 이동 명령 수신")
                    text = (f"네, following 기능을 수행하겠습니다.")
                    # TTS로 장소 이동 응답
                    asyncio.run(tts.speak(text))
                    # 서비스 클라이언트 실행: /pickee/mobile/change_tracking_mode (mode='tracking')
                    ros.service_call(robot_id=1,mode="tracking",timeout_sec=15.0)
                # 그외 답변이면
                else:
                    print("[LLM] 등록 외 명령어 수신")

        # 사용자가 ctrl+c를 누르면 종료 안내 후 루프 종료
        except KeyboardInterrupt:
            print("종료")
            ros.shutdown()
            break
        # 에러 발생 시 출력 후 다음 루프로 진행
        except Exception as e:
            print(f"에러 발생 : {e}")
            continue

# 메인함수 실행
if __name__ =='__main__':
    main()

