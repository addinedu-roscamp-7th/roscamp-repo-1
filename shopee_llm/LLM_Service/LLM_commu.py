# ----------------------- tool calling 관련 라이브러리 선언 -----------------------
from langgraph.prebuilt import create_react_agent
# langchain에서 ollama 사용을 위한 ChatOllama 라이브러리 import
from langchain_ollama import ChatOllama
# langchain tool 사용을 위한 tool 라이브러리 import
from langchain.tools import Tool
# 툴 사용 여부 판별을 위한 ToolMessage, AIMessage 라이브러리 import
from langchain_core.messages import ToolMessage, AIMessage
# ----------------------- tool calling 관련 라이브러리 선언 -----------------------
from langgraph.prebuilt import create_react_agent
# langchain에서 ollama 사용을 위한 ChatOllama 라이브러리 import
from langchain_ollama import ChatOllama
# ------------------------------ 커스텀 클래스 import ----------------------------
from HTTP_commu import HTTP_Util

# LLM에서 사용할 Tool class 선언
class Tool_function():
    def __init__(self):
        #llm 사용 모델 선언 
        self.model_name = "llama3.1"
        # HTTP_Util 객체 선언
        self.data_handler = HTTP_Util()
    # tool에 적용할 상품 정보 관련 함수 선언
    def item_info(self,text):
        # 에러가 없으면
        try:
            # llm에 요청 항목
            request = (
                "역할: 한국어 정보 요청 명령에서 상품 이름만 산출.\n"
                "규칙1: '사과 관련 정보 알려줘' → 사과, '닭에 관해서 알고 싶어' → 닭, '사탕 알려줘' → 사탕\n"
                "규칙2: 출력은 오직 상품 이름 한개만. 공백/문장/단위/ 금지. 예) 닭, 전복,생선,사탕,음료수"
            )
            # llm 객체 설정
            # llama 3.1 모델 사용
            llm = ChatOllama(model=self.model_name, temperature=0)
            # 프롬프트 제작
            # 마지막에 /n응답을 붙여주면 llm이 딱 필요한 응답만 내뱉음
            prompt = f"{request}\n\n사용자 입력: {text}\n응답:"
            # prompt에 저장된 문자열을 llm에 입력
            answer = llm.invoke(prompt)
            # llm응답이 문자열(str)이면 그대로 raw_result에 저장하고 메시지 객체면 content(내용)만 추출해서 저장
            # llm이 문자열을 줄 수도 있고 메시지 객체를 줄 수도 있음 -> 둘중에 어는 걸로 답변해줘도 답변만 저장
            # getattr : (객체, 속성 이름, 기본값)으로 객체에서 해당 속성만 꺼내오는 함수
            # x if () else y : 조건이 참이면 x를 아니면 y를 추출
            result = answer if isinstance(answer, str) else getattr(answer, "content", "")
            # llm 답변에서 앞뒤 공백 제거하거나 1,000 등에서 ,  제거
            result = (result or "").strip().replace(",", "")
            print("----------------------------------------")
            print(f"[LLM_Service] 사용자 정보 요청 상품: {result} ")
            print("----------------------------------------")
            # 상품 정보 메시지 제작
            item_info_message = self.data_handler.make_search_query_message(result)
            # llm 반환값 return
            return item_info_message
        # 에러가 발생하면
        except Exception as e:
            # 에러 print
            print(f"[LLM_Service] 예외 발생: {e}")
            # 0 return
            return 0
    
    # tool에 적용할 상품 픽업 관련 함수 선언
    def pickup_info(self,text):
        # 에러가 없으면
        try:
            # llm에 요청 항목
            request = (
                "역할: 한국어 상품 선택 명령에서 번호만 산출.\n"
                "규칙1: '2번 집어' → 2, '5번 집어줘/골라줘' → 5, '17번' → 17, '13번 골라줄래?→ 13, '5번 골라 → 5'\n"
                "규칙2: 출력은 오직 상품 번호 한개만. 공백/문장/단위 금지. 예) 닭, 전복,생선,사탕,음료수"
            )
            # llm 객체 설정
            # llama 3.1 모델 사용
            llm = ChatOllama(model=self.model_name, temperature=0)
            # 프롬프트 제작
            # 마지막에 /n응답을 붙여주면 llm이 딱 필요한 응답만 내뱉음
            prompt = f"{request}\n\n사용자 입력: {text}\n응답:"
            # prompt에 저장된 문자열을 llm에 입력
            answer = llm.invoke(prompt)
            # llm응답이 문자열(str)이면 그대로 raw_result에 저장하고 메시지 객체면 content(내용)만 추출해서 저장
            # llm이 문자열을 줄 수도 있고 메시지 객체를 줄 수도 있음 -> 둘중에 어는 걸로 답변해줘도 답변만 저장
            # getattr : (객체, 속성 이름, 기본값)으로 객체에서 해당 속성만 꺼내오는 함수
            # x if () else y : 조건이 참이면 x를 아니면 y를 추출
            result = answer if isinstance(answer, str) else getattr(answer, "content", "")
            # llm 답변에서 앞뒤 공백 제거하거나 1,000 등에서 ,  제거
            result = (result or "").strip().replace(",", "")
            # 정수형태로 별환
            result = int(result)
            print("--------------------------------------------------")
            print(f"[LLM_Service] 사용자 픽업 요청 상품: {result} ")
            print("--------------------------------------------------")
            # 픽업 상품 메시지 제작
            pickup_info_message = self.data_handler.make_bbox_message(result)
            # llm 반환값 return
            return pickup_info_message
        # 에러가 발생하면
        except Exception as e:
            # 에러 print
            print(f"[LLM_Service] 예외 발생: {e}")
            # 0 return
            return 0
        
    # tool에 적용할 상품 픽업 관련 함수 선언
    def move_info(self,text):
        # 에러가 없으면
        try:
            # llm에 요청 항목
            request = (
                "역할: 한국어 이동 명령에서 이동 장소만 산출.\n"
                "규칙1: '반납대로 이동해' → 반납대, '신선식품으로 이동해' → 신선식품, '기성품 코너로 가봐' → 기성품, '포장대로 가'→ 포장대, '반납대 가줘' → 반납대'\n"
                "규칙2: 출력은 오직 이동 장소 한개만. 공백/문장/단위/ 금지. 예) 닭, 전복,생선,사탕,음료수"
            )
            # llm 객체 설정
            # llama 3.1 모델 사용
            llm = ChatOllama(model=self.model_name, temperature=0)
            # 프롬프트 제작
            # 마지막에 /n응답을 붙여주면 llm이 딱 필요한 응답만 내뱉음
            prompt = f"{request}\n\n사용자 입력: {text}\n응답:"
            # prompt에 저장된 문자열을 llm에 입력
            answer = llm.invoke(prompt)
            # llm응답이 문자열(str)이면 그대로 raw_result에 저장하고 메시지 객체면 content(내용)만 추출해서 저장
            # llm이 문자열을 줄 수도 있고 메시지 객체를 줄 수도 있음 -> 둘중에 어는 걸로 답변해줘도 답변만 저장
            # getattr : (객체, 속성 이름, 기본값)으로 객체에서 해당 속성만 꺼내오는 함수
            # x if () else y : 조건이 참이면 x를 아니면 y를 추출
            result = answer if isinstance(answer, str) else getattr(answer, "content", "")
            # llm 답변에서 앞뒤 공백 제거하거나 1,000 등에서 ,  제거
            result = (result or "").strip().replace(",", "")
            # 정수형태로 별환
            result = int(result)
            print("--------------------------------------------------")
            print(f"[LLM_Service] 사용자 이동 요청 장소: {result} ")
            print("--------------------------------------------------")
            # 픽업 상품 메시지 제작
            move_info_message = self.data_handler.make_intent_detection_message(result)
            # llm 반환값 return
            return move_info_message
        # 에러가 발생하면
        except Exception as e:
            # 에러 print
            print(f"[LLM_Service] 예외 발생: {e}")
            # 0 return
            return 0

# llama 3.1에 입력을 하면 답을 밷는 클래스 선언 
class LLM_response():
    def __init__(self):
        #llm 사용 모델 선언 
        self.model_name = "llama3.1"
        # llm에 지시사항으로 넣어줄 문구
        self.instructions = (
            "You are a helpful assistant. "
            "For greetings, weather, chit-chat, or anything else, answer directly WITHOUT tools. "
            "you MUST call the 'parsing_goal' tool to extract the signed distance. "
            "You must say only Korean "
            "Do NOT write meta comments like 'Note:'—just give the answer. ")
        # Tool_function 클래스 객체 생성
        self.tool_function = Tool_function()

    # llm에 사용자 텍스트화된 음성 적용
    def get_llm_reponse(self,text):
        tools = [
            Tool(name='item_info'  , func=self.tool_function.item_info,   description="상품 정보 문의 관련 요청에만 답변"),
            Tool(name='pickup_info', func=self.tool_function.pickup_info, description="상품 선택 관련 요청에만 답변"),
            Tool(name='move_info'  , func=self.tool_function.move_info,   description="장소가 포함된 이동 명령 관련 요청에만 답변")]
        # llm 객체 설정
        # llama3.1 모델 사용
        llm = ChatOllama(model="llama3.1")
        # langchain 객체 활용하여 질문에 반응하는 agent 객체 생성
        agent = create_react_agent(llm,tools)
        # agent에 question에 저장된 질문 입력 후 answer에 응답 저장
        answer = agent.invoke({"messages": [("system", self.instructions),("user", text)]})
        # LLM 응답을 확인
        msgs = answer["messages"]
        # LLM 응답에 ToolMessage가 하나라도 있거나 LLM이 요청한 메시지 중 tool_calls 형식의 
        # 메시지가 하나라도 있으면 used_tool True로 설정
        used_tool = (any(isinstance(m,ToolMessage) for m in msgs) 
                    or any(isinstance(m, AIMessage)and getattr(m,"tool_calls",None) for m in msgs))
        # tool이 실행되지 않았으면
        if not used_tool:
            # tool이 아닌 llm 답변 출력
            print(answer["messages"][-1].content)