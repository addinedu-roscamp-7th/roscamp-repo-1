# ------------------------------ HTTP 관련 라이브러리 import ------------------------------
# HTTP 요청을 보낼 수 있는 requests 라이브러리 import
import requests

# 메인 함수 선언
def main():
    while True:
        try:
            print("---사용자 음성 의도 분석 ---")
            # 사용자 입력
            user_input = input()

            if user_input == '1':
                # Flask 서버의 /hello 경로로 GET 요청을 전송
                # URL : http://192.168.0.154:5001/llm/search_query
                # 서버가 응답을 반환하면 response에 저장
                response = requests.get("http://192.168.0.154:5001/llm/search_query")
                # response를 json 형식으로 변환하여 출력
                print(response.json())

            if user_input == '2':
                # Flask 서버의 /hello 경로로 GET 요청을 전송
                # URL : http://192.168.0.154:5001/llm/bbox
                # 서버가 응답을 반환하면 response에 저장
                response = requests.get("http://192.168.0.154:5001/llm/bbox")
                # response를 json 형식으로 변환하여 출력
                print(response.json())

            if user_input == '3':
                # Flask 서버의 /hello 경로로 GET 요청을 전송
                # URL : http://192.168.0.154:5001/llm/intent_detection
                # 서버가 응답을 반환하면 response에 저장
                response = requests.get("http://192.168.0.154:5001/llm/intent_detection")
                # response를 json 형식으로 변환하여 출력
                print(response.json())
        # 사용자가 ctrl+c를 누르면 종료 안내 후 루프 종료
        except KeyboardInterrupt:
            print("종료")
            break


if __name__ =='__main__':
    main()