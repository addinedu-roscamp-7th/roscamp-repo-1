# ------------------------------ 라이브러리 import ------------------------------
# Python 딕셔너리를 JSON 형식으로 변환해 응답으로 보내기 위한 jsonify 라이브러리 import
from flask import jsonify
# HTTP 요청 파라미터 접근을 위한 request 라이브러리 import
from flask import request

class HTTP_Util():
    def __init__(self):
        print("")

    def data_to_json(self,data):
        print(f"[LLM_Service]: send {data}")
        return jsonify(data)
    
    def make_search_query_message(self,data):
        message = {"sql_query": f"name LIKE '%{data}%'"}
        return message
    
    def make_bbox_message(self,data):
        message = {"bbox": f"{data}"}
        return message
    
    def make_intent_detection_message(self,data):
        message =  {
        "intent": "Move_place",
        "entities":{
            "place_name" : f"{data}",
            "action" : "move"}}
        return message
    
    def get_data_from_request(self):
        # 데이터 저장 변수
        data = None
        # GET method에서 text를 추출
        if request.method == "GET":
            data = request.args.get("text",type=str)
        # 데이터 공백 정리
        data = (data or "").strip()
        return data if data else "없음"