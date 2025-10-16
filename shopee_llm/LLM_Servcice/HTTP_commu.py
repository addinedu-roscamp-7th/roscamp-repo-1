# ------------------------------ 라이브러리 import ------------------------------
# Python 딕셔너리를 JSON 형식으로 변환해 응답으로 보내기 위한 jsonify 라이브러리 import
from flask import jsonify

class HTTP_Util():
    def __init__(self):
        print("----- Pickee LLM Service -----")
    def data_to_json(self,data):
        print(f"[LLM_Service]: send {data}")
        return jsonify(data)