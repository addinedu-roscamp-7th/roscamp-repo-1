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
        # Tool 함수 결과 저장 변수 선언
        self.tool_result = None
    # tool에 적용할 상품 정보 관련 함수 선언
    def item_info(self,text):
        # 에러가 없으면
        try:
            # llm에 요청 항목
            request = (
                "[역할]\n"
                    "- 너는 한국어 사용자의 질의에서 ‘상품 이름’ 1개만 추출하는 추출기다.\n"
                    "- 대화/설명/사과문/메타 코멘트 금지. 정답만 출력한다.\n\n"
                "[추출 규칙]\n"
                    "1) 결과는 ‘상품 이름’ 1개(일반 명사). 예) 사과, 닭, 전복, 생선, 사탕, 음료수\n"
                    "2) 금지: 문장, 설명, 따옴표, 마침표, 특수문자, 이모지, 단위(kg, g, 개, 봉 등), 수식어.\n"
                    "3) 수식어 제거: “맛있는 초콜릿” → 초콜릿, “국산 사과” → 사과\n"
                    "4) 수량/가격/순번 무시: “3번 사과”, “사과 2개 3000원” → 사과\n"
                    "5) 여러 후보 등장 시 핵심 토픽 1개만: “사과랑 배 중에 사과 알려줘” → 사과\n"
                    "6) 모호하거나 상품이 아니면 “없음” 출력.\n"
                    "7) 외래어/브랜드/카테고리 혼재 시 보편적 상품 일반명으로 정규화:\n"
                        "   - “코카콜라” → 코카콜라 (브랜드가 곧 상품명인 경우 허용)\n"
                        "   - “탄산음료 알려줘” → 탄산음료 (카테고리도 상품명으로 인정)\n"
                    "8) 금칙 패턴: 줄바꿈, 콜론(:), 따옴표(\"'), 괄호(), 마크다운, 설명문.\n\n"
                        "[예시]\n"
                        "- 입력: \"사과 관련 정보 알려줘\" → 출력: 사과\n"
                        "- 입력: \"닭에 관해서 알고 싶어\" → 출력: 닭\n"
                        "- 입력: \"사탕 알려줘\" → 출력: 사탕\n"
                        "- 입력: \"전복이랑 생선 중에 뭐가 좋아?\" → 출력: 전복  (또는 생선 중 핵심을 하나만)\n"
                        "- 입력: \"가격 알려줘\" → 출력: 없음\n"
                        "- 입력: \"5번 초콜릿 보여줘\" → 출력: 초콜릿\n\n"
                "[출력 형식]\n"
                    "- 한 줄, 한 단어(또는 일반 명사구 1개)만 출력.\n"
                    "- 예: 사과")
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
            self.tool_result = item_info_message
            # llm 반환값 return
            return item_info_message
        # 에러가 발생하면
        except Exception as e:
            # 에러 print
            print(f"[LLM_Service] 예외 발생: {e}")
            # 0 return
            return self.tool_result
    
    # tool에 적용할 상품 픽업 관련 함수 선언
    def pickup_info(self,text):
        # 에러가 없으면
        try:
            # llm에 요청 항목
            request = (
                "[역할]\n"
                    "- 너는 한국어 '상품 선택' 명령에서 **선택 번호** 1개만 추출한다.\n"
                    "- 대화/설명/사과/메타 코멘트 금지. **정답만 숫자**로 출력한다.\n\n"
                "[추출 규칙]\n"
                    "1) 결과는 **아라비아 숫자 정수 1개**. 예) 2, 5, 17, 103\n"
                    "2) 금지: 문장, 단어, 기호, 단위(개/원/% 등), 따옴표, 마침표, 줄바꿈.\n"
                    "3) 한국어 숫자어(한글 표기)도 아라비아 숫자로 변환: 다섯→5, 열일곱→17.\n"
                    "4) 다수의 숫자가 있을 때 **'번' 앞에 오는 숫자**를 최우선으로 선택.\n"
                    "   - 예: '3번이랑 5번 중에 5번 골라' → 5\n"
                    "   - 예: '5번 집어주고 2개 담아' → 5 (수량/가격 등은 무시)\n"
                    "5) '선택/골라/집어/픽업/번호/번' 등 **선택 의도 키워드**가 있으면 그 맥락의 숫자만 고려.\n"
                    "6) 가격/수량/거리/시간 등 **선택과 무관한 숫자**는 무시: '사과 2개 3000원 중 1번' → 1\n"
                    "7) 적절한 선택 번호가 **전혀 없으면 0**만 출력.\n\n"
                    "[예시]\n"
                        "- 입력: '2번 집어' → 출력: 2\n"
                        "- 입력: '5번 골라줘' → 출력: 5\n"
                        "- 입력: '17번' → 출력: 17\n"
                        "- 입력: '3번이랑 5번 중 하나 골라' → 출력: 5 (가장 최근 선택 동사와 결합)\n"
                        "- 입력: '사과 2개 3000원인데 1번으로' → 출력: 1\n"
                        "- 입력: '반납대로 이동해' → 출력: 0 (선택 번호 없음)\n\n"
                "[출력 형식]\n"
                    "- **숫자만** 한 줄로 출력 (예: 5)\n")
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
            self.tool_result = pickup_info_message
            # llm 반환값 return
            return pickup_info_message
        # 에러가 발생하면
        except Exception as e:
            # 에러 print
            print(f"[LLM_Service] 예외 발생: {e}")
            # 0 return
            return self.tool_result
        
    # tool에 적용할 상품 픽업 관련 함수 선언
    def move_info(self,text):
        # 에러가 없으면
        try:
            # llm에 요청 항목
            request = (
                "[역할]\n"
                    "- 당신은 한국어 '이동' 명령에서 **이동 장소명** 1개만 추출한다.\n"
                    "- 대화/설명/사과/메타 코멘트 금지. **정답(장소명)만** 출력한다.\n\n"
                "[추출 규칙]\n"
                    "1) 결과는 **장소명 1개**(일반 명사/명사구). 예) 반납대, 신선식품, 포장대, 계산대, 고객센터, 창고, 입구\n"
                    "2) 결과는 무조건 '반납함, 신선식품, 포장대, 계산대' 4개중 1개만 올 수 있음 \n"
                    "3) 금지: 문장, 동사, 조사, 따옴표, 기호, 마침표, 줄바꿈, 수량/가격/거리 등 단위.\n"
                    "4) 동사/조사/수식어 제거: '반납대로 이동해' → 반납대, '신선식품 코너로 가줘' → 신선식품\n"
                    "5) 동의어/변형 정규화:\n"
                    "   - 반납함/반납코너/반납대로/반납장 → 반납함\n"
                    "   - 신선/신선코너/냉장식품/생식품 → 신선식품\n"
                    "   - 포장대로/포장 코너/포장 공간 → 포장대\n"
                    "   - 계산대/카운터/결제대 → 계산대\n"
                    "6) 여러 장소가 동시에 등장하면 **이동 동사(가/가줘/가봐/이동/가라/이동해/향해 등)**와 가장 가깝게 결합된\n"
                    "   마지막 목적지를 선택. 예) '신선식품 들렀다가 포장대로 가' → 포장대\n"
                    "7) 숫자/순번/가격/수량은 무시. 예) '3번 코너에 있는 포장대로 가' → 포장대\n"
                    "8) 방향/거리만 있고 장소가 없으면 **'없음'** 출력. 예) '앞으로 1미터 가' → 없음\n"
                    "9) 장소가 애매하거나 없는 경우에도 **'없음'**만 출력.\n\n"
                        "[예시]\n"
                        "- 입력: '반납대로 이동해' → 출력: 반납대\n"
                        "- 입력: '신선식품으로 가줘' → 출력: 신선식품\n"
                        "- 입력: '기성품 코너로 가봐' → 출력: 기성품\n"
                        "- 입력: '포장대로 가' → 출력: 포장대\n"
                        "- 입력: '계산대 말고 포장대로 가줘' → 출력: 포장대\n"
                        "- 입력: '입구 쪽으로 이동' → 출력: 입구\n"
                        "- 입력: '앞으로 1미터 전진해' → 출력: 없음\n\n"
                    "10) 2)번 규칙은 꼭 지켜져야함"
                "[출력 형식]\n"
                    "- 한 줄, **장소명만** 출력 (예: 포장대)\n")
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
            print("--------------------------------------------------")
            print(f"[LLM_Service] 사용자 이동 요청 장소: {result} ")
            print("--------------------------------------------------")
            # 픽업 상품 메시지 제작
            move_info_message = self.data_handler.make_intent_detection_message(result)
            self.tool_result = move_info_message
            # llm 반환값 return
            return move_info_message
        # 에러가 발생하면
        except Exception as e:
            # 에러 print
            print(f"[LLM_Service] 예외 발생: {e}")
            # 0 return
            return self.tool_result

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
            "Do NOT write meta comments like 'Note:'—just give the answer. "
            "도구 선택 규칙: 숫자 선택 요청은 'pickup_info', 장소 이동은 'move_info', "
            "일반 상품 질의는 'item_info'를 **정확히 하나만** 사용하라. "
            "pickup_info를 사용할 때는 출력은 숫자만 반환해야 한다.")
        # Tool_function 클래스 객체 생성
        self.tool_function = Tool_function()

    # llm에 사용자 텍스트화된 음성 적용
    def get_llm_reponse(self,text):
        tools = [
            Tool(name='item_info'  , 
                 func=self.tool_function.item_info,   
                 description=("상품 정보 문의일 때만 사용. 키워드: '정보', '알려줘', '에 관해', '설명'. "
                             "숫자 선택(예: 3번)이나 이동 장소(반납대/신선식품/포장대 등)가 포함되면 사용 금지. ")
                ),
            Tool(name='pickup_info', 
                 func=self.tool_function.pickup_info, 
                 description=("선택 번호 요청일 때만 사용. 키워드: 'N번', '집어', '골라'. "
                             "입력에서 **숫자**만 존재해야 하며 장소 단어(반납대, 신선식품 등)가 있으면 사용 금지.")
                 ),
            Tool(name='move_info'  , 
                 func=self.tool_function.move_info,   
                 description=("장소 이동 요청일 때만 사용. 키워드: '이동', '가', '가줘', '가봐', '로 이동', "
                               "장소 단어(반납대, 신선식품, 포장대, 기성품 등)가 있을 때 사용. 숫자 선택 요청이면 사용 금지.")
                 )]
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
        if used_tool:
            # 툴이 실행되었으면 tool의 마지막 답변을 반환해주거나 데이터가 없다고 반환
            return self.tool_function.tool_result  or {"status": "ok", "detail": "tool executed but no payload"}
        # tool이 실행되지 않았으면
        if not used_tool:
            # tool이 아닌 llm 답변 반한
            content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            return {"answer": (content or "").strip()}