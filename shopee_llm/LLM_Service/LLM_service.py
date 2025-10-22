# ------------------------------ HTTP 관련 라이브러리 import ------------------------------
# http 객체 생성을 위한 Flask 라이브러리 import
from flask import Flask
# Python 딕셔너리를 JSON 형식으로 변환해 응답으로 보내기 위한 jsonify 라이브러리 import
from flask import jsonify
# HTTP Util 클래스 import
from HTTP_commu import HTTP_Util

# ---------------------------------- 커스텀 클래스 import --------------------------------
# Tool 함수가 선언된 Tool_function class import
from LLM_commu import LLM_response
# ---------------------------------- Flask 객체 생성 ------------------------------------
# Flask 객체 생성
# __name__ : 현재 실행 중인 모듈의 이름 (현재 프로그램 이름 자동으로 넣어줌)
app = Flask(__name__)
# 데이터를 제이슨 변환을 위한 객체 생성
llm_handler = LLM_response()
# 요청에서 데이터 추출을 위한 객체 생성
http_handler = HTTP_Util()
# ------------------------------ 상품 정보 요청 처리  ------------------------------
# 클라이언트에서 /llm/search_query 경로로 GET 요청이 들어오면 app.route에 등록된 함수가 실행
@app.route('/llm/search_query', methods=['GET'])
# app.route 조건이 충족되었을 때 실행시킬 search_query 함수
def search_query():
    print("상품 정보 요청 수신")
    mocup_data = "닭에 관련한 정보 알려줘"
    receive_data = http_handler.get_data_from_request()
    search_query_messange = llm_handler.get_llm_reponse(receive_data)
    return jsonify(search_query_messange)
# ------------------------------ 상품 픽업 요청 처리  ------------------------------
# 클라이언트에서 /llm/bbox 경로로 GET 요청이 들어오면 app.route에 등록된 함수가 실행
@app.route('/llm/bbox', methods=['GET'])
# app.route 조건이 충족되었을 때 실행시킬 bbox 함수
def bbox():
    print("상품 픽업 요청 수신")
    mocup_data = "18번 골라"
    receive_data = http_handler.get_data_from_request()
    pickup_message = llm_handler.get_llm_reponse(receive_data)
    return jsonify(pickup_message)
# ------------------------------ 발화 의도 분석 ------------------------------
# 클라이언트에서 /llm/intent_detection 경로로 GET 요청이 들어오면 app.route에 등록된 함수가 실행
@app.route('/llm/intent_detection', methods=['GET'])
# app.route 조건이 충족되었을 때 실행시킬 intent_dectection 함수
def intent_detection():
    print("사용자 이동 명령 요청 수신")
    mocup_data = "신선식품 매대로 가줘"
    receive_data = http_handler.get_data_from_request()
    intent_message = llm_handler.get_llm_reponse(receive_data)
    return jsonify(intent_message)
# ------------------------------ 서버 실행 ------------------------------
# port 5000번으로 클라이언트 실행
app.run(host="0.0.0.0",port=5001)