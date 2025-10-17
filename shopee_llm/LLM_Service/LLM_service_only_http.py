# ------------------------------ 라이브러리 import ------------------------------
# http 객체 생성을 위한 Flask 라이브러리 import
from flask import Flask
# HTTP_commu.py에 정의된 HTTP_commu class import
from HTTP_commu import HTTP_Util
# ------------------------------ Flask 객체 생성 ------------------------------
# Flask 객체 생성
# __name__ : 현재 실행 중인 모듈의 이름 (현재 프로그램 이름 자동으로 넣어줌)
app = Flask(__name__)
# 데이터를 제이슨 변환을 위한 객체 개성
data_handler = HTTP_Util()
# ------------------------------ 상품 정보 요청 처리  ------------------------------
# 클라이언트에서 /llm/search_query 경로로 GET 요청이 들어오면 app.route에 등록된 함수가 실행
@app.route('/llm/search_query', methods=['GET'])
# app.route 조건이 충족되었을 때 실행시킬 search_query 함수
def search_query():
    print("상품 정보 요청 수신")
    data = {"sql_query": "name LIKE '%사과%'"}
    query_data = data_handler.data_to_json(data)
    return query_data   
# ------------------------------ 상품 픽업 요청 처리  ------------------------------
# 클라이언트에서 /llm/bbox 경로로 GET 요청이 들어오면 app.route에 등록된 함수가 실행
@app.route('/llm/bbox', methods=['GET'])
# app.route 조건이 충족되었을 때 실행시킬 bbox 함수
def bbox():
    print("상품 픽업 요청 수신")
    data = {"bbox": "2"}
    pickup_data = data_handler.data_to_json(data)
    return pickup_data
# ------------------------------ 발화 의도 분석 ------------------------------
# 클라이언트에서 /llm/intent_detection 경로로 GET 요청이 들어오면 app.route에 등록된 함수가 실행
@app.route('/llm/intent_detection', methods=['GET'])
# app.route 조건이 충족되었을 때 실행시킬 intent_dectection 함수
def intent_detection():
    print("사용자 이동 명령 요청 수신")
    data = {
        "intent": "Move_place",
        "entities":{
            "place_name" : "반납함",
            "action" : "move"}}
    intent_data = data_handler.data_to_json(data)
    return intent_data
# ------------------------------ 서버 실행 ------------------------------
# port 5000번으로 서버 실행
app.run(host="192.168.0.154",port=5001)