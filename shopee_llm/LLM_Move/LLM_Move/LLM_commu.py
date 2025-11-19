# --------------------------------LLM 관련 라이브러리 import --------------------------------
# OpenAI 호환(vllm)을 위한 ChatOpenAI import
from langchain_openai import ChatOpenAI
# chat template 적용을 위한 ChatPromptTemplate import
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# langchain tool 에이전트 호출을 위한 create_tool_calling_agent import
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
# 에이전트 실행을 위한 AgentExecutor import
from langchain.agents import AgentExecutor
# langchain tool 사용을 위한 tool 라이브러리 import
from langchain.tools import Tool

class LLM_Util():
    # 클래스 초기화 함수 선언
    def __init__(self):
        self.model = "/home/addinedu/LLM/LLM_Tuing_Model/tunning_model"
    # --------- vLLM의 OpenAI 호환 서버에 Tool에서 사용할 LLM 연동을 위한 LLM 래퍼 설정 ---------------------
    def tool_llm(self):
        # vLLM의 OpenAI 호환 서버에 Tool에서 사용할 LLM 연동을 위한 LLM 래퍼 설정
        tool_use_llm = ChatOpenAI(
                                # 모델 경로 설정
                                model= self.model,
                                # local 주소 설정
                                base_url= "http://localhost:8000/v1",
                                # api key는 사용 안함
                                api_key="EMPTY",
                                # llm의 일관성을 높히기 위해 temperature는 낮게 설정
                                temperature=0.3)
        # llm 래퍼 반환
        return tool_use_llm
    # --------- # vLLM의 OpenAI 호환 서버에 LLM을 연동을 위한 LLM 래퍼 설정---------------------
    def main_llm(self):
        # vLLM의 OpenAI 호환 서버에 LLM을 연동을 위한 LLM 래퍼 설정
        main_use_llm = ChatOpenAI(
                                # 모델 경로 설정
                                model= self.model,
                                # local 주소 설정
                                base_url= "http://localhost:8000/v1",
                                # api key는 사용 안함
                                api_key="EMPTY",
                                # llm의 일관성을 높히기 위해 temperature는 낮게 설정
                                temperature=0.3,
                                # vLLM에 툴 활성화 및 한 응답에서 툴 1개만 사용 설정
                                model_kwargs={"tool_choice": "required",
                                              "parallel_tool_calls": False})
        # llm 래퍼 반환
        return main_use_llm        
    # --------- LLM에 응답을 넣어주면 답변만 반환하는 함수 선언 ---------------------
    def tool_llm_response(self,llm,prompt):
        # llm에 prompt를 입력하면 응답을 tool_use_llm에 입력
        tool_llm_answer = llm.invoke(prompt)
        # tool_llm_answer가 str 형식이면 바로 tool_llm_response에 저장하고 메시지 형태이면 "content"를 저장하고 없으면 ""를 저장
        tool_llm_response = tool_llm_answer if isinstance(tool_llm_answer, str) else getattr(tool_llm_answer, "content", "") 
        # llm답변만 공백 제거후 return
        return (tool_llm_response or "").strip()
    
class Tool_Util():
    def __init__(self):
        # LLM_Util 객체 활용을 위한 객체 선언
        self.llm_util = LLM_Util()
    # 상품 정보를 요청하면 mockup 재고 데이터로 응답하는 간단한 item_info 함수 선언
    def item_info(self,text):
        # mockup 답변 설정
        mockup_answer = "재고는 2개 입니다."
        # mockup 답변 return
        return mockup_answer
    # 상품 위치를 요청하면 mockup 좌표를 응답하는 간단한 item_coordinate 함수 선언
    def item_coordinate(self,text):
        # mockup 답변 설정
        mockup_answer = "상품 위치는 x:100,y:21,z:0 입니다."
        # mockup 답변 return
        return mockup_answer
    # 장소를 기반으로 한 장소 이동 요청을 하였을 때 LLM을 사용하여 파싱하여 return할 함수 선언 
    def move_place(self,text):
        # llm에 입력할 프롬프트 규칙을 알려주는 지시문 설정
        prompt = (
            "역할: 한국어 장소 이동 명령에서 목적지 장소명 1개만 추출 \n"
            "규칙:\n"
            "1) 결과는 명사/명사구 1개(조사/따옴표/기호/마침표 제거)\n"
            "2) '~으로/~로/~까지' 등 조사는 삭제.\n"
            "3) 장소가 없거나 애매하면 '없음'만 출력.\n"
            "4) 응답은 딱 한 줄, 장소명만 출력.\n"
            "예시: '포장대로 가줘'→포장대, '신선식품 코너로 이동'→신선식품, '과자로 가줘'→과자\n"
            "후보 {반납함, 신선식품, 기성품, 과자, 포장대, 계산대, 없음} 중 하나만 정확히 출력."
            "금지: 함수명, JSON, 중괄호, 따옴표, 설명, 접두/접미 텍스트."
            "출력 예시: 기성품"
            f"\n사용자 입력: {text}\n응답:"
        )
        # tool용 llm 래퍼 저장
        tool_use_llm = self.llm_util.tool_llm()
        # prompt를 입력하여 llm에서 응답 받기
        llm_response = self.llm_util.tool_llm_response(tool_use_llm,prompt)
        # llm 응답 return
        return str(llm_response)
    # 팔로잉 요청을 하였을 때 LLM을 사용하여 following을 return할 함수 선언 
    def following_user(self,text):
        # llm에 입력할 프롬프트 규칙을 알려주는 지시문 설정
        prompt = (
            "역할: 사용자의 한국어/영어 문장이 '팔로우/따라오기/이리 오기' 명령인지 판별하는 분류기.\n"
            "출력 형식: 아래 중 하나만 한 줄로 출력\n"
            "- following : '따라와/이리 와/옆으로 와/가까이 와/Follow me/Come here'류 명령으로 해석될 때\n"
            "- 없음 : 그 외(질문/서술/농담/부정/금지/다른 의도)\n"
            "규칙:\n"
            "1) 공백·이모지·문장부호·반복 글자(와아아 등)는 무시해 의미만 판단.\n"
            "2) 정중체/반말/명령/부탁/의문형(예: '~해줄래?', '~와?') 모두 허용.\n"
            "3) 다음과 같은 부정 표현이 포함되면 무조건 '없음': "
            "'오지 마', '따라오지 마', '오지 말고', '따라오지 말고', '오면 안', '오지 않아', "
            "'오지마세요', \"don't\", \"do not\", 'no'.\n"
            "4) 예시 판단:\n"
            "   - '따라와', '따라와줘', '뒤따라와', '나 따라와', '붙어 있어', '내 옆으로 와', '앞으로 와', "
            "'가까이 와', '여기로 와', '이리 와', '일로 와' → following\n"
            "   - '따라오지 마', '오지 마', '따라오지 말고 거기 있어' → 없음\n"
            "   - 'follow me', 'come here', 'stay close', 'stick with me' → following\n"
            "금지: 설명/부가 텍스트/따옴표/기호/JSON 출력. 정답 단어만 출력.\n"
            f"\n사용자 입력: {text}\n응답:"
        )
        # tool용 llm 래퍼 저장
        tool_use_llm = self.llm_util.tool_llm()
        # prompt를 입력하여 llm에서 응답 받기
        llm_response = self.llm_util.tool_llm_response(tool_use_llm,prompt)
        # llm 응답 return
        return str(llm_response)

    # tool 관련 정보 반환 함수 선언
    def tool_info(self):
            tools = [
            Tool(
                name="item_info",
                func=self.item_info,
                description="상품/대상의 재고 관련 질문일 때만 사용.",
                return_direct=True
            ),
            Tool(
                name="item_coordinate",
                func=self.item_coordinate,
                description="상품/대상의 위치/좌표를 물을 때만 사용.",
                return_direct=True
            ),
            Tool(
                name="move_place",
                func=self.move_place,
                description="‘~로/~으로/~까지’ 등 목적지 표현이 있을 때 장소명 1개만 추출.",
                return_direct=True
            ),
            Tool(
                name="following_user",
                func=self.following_user,
                description="'따라와' 등 추종 명령이 있을 때 following을 반환",
                return_direct=True
            )]
            return tools
    
class GET_LLM_Response():
    def __init__(self):
        # LLM_Util 객체 선언
        self.llm_util = LLM_Util()
        # Tool_Util 객체 선언
        self.tool_util = Tool_Util()
        # 적용할 prompt 설정
        self.prompt = ChatPromptTemplate.from_messages([
                        # system으로 모델에게 시스템에서 적용하야할 지시문을 작성
                        ("system",
                         "한국어만 사용.\n"
                         "도구 사용 규칙:\n"
                         "- 재고/수량 문의 → item_info\n"
                         "- 상품 위치/좌표 문의 → item_coordinate\n"
                         "- '~로/~으로/~까지' 등 목적지가 포함된 장소 이동 → move_place\n"
                         "도구 이름은 반드시 다음 중 하나만 정확히 사용할 것: item_info, item_coordinate, move_place,following_user \n"
                         "moveplace/iteminfo/itemcoordinate/followinguser 같은 오탈자를 만들지 말 것.\n"
                         "도구를 사용했다면, 도구가 반환한 문자열을 그대로 사용자에게 출력\n"
                         "도구가 불필요하면 간단명료하게 답하라."
                         "도구는 반드시 OpenAI Tools 호출로만 사용하고, 함수명+JSON을 텍스트로 출력하지 말 것."
                         "도구를 사용했다면, 도구가 반환한 문자열을 그대로 사용자에게 출력."
                        ),  
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad")])
        
    def llm_answer(self,input):
        # main_llm 래퍼 저장
        main_llm = self.llm_util.main_llm()
        # tool 설정 저장
        tools = self.tool_util.tool_info()
        # OpenAI 툴 규약을 따르는 LangChain 에이전트 생성
        agent = create_tool_calling_agent(main_llm,tools,self.prompt)
        # 생성된 에이전트를 실행가능하도록 랩핑
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        # 에이전트에 입력을 전달하고 실행
        llm_response = executor.invoke({"input": input})
        # 답변만 저장
        llm_out = llm_response["output"]
        # 디버그 print
        # print("--- LLM 답변 ---")
        # print(f"[LLM] {llm_out}")
        return llm_out