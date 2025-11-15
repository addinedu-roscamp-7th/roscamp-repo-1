# json 활용을 위해 json import
import json
# lanchain toll 사용을 위해 tool import
from langchain.tools import tool
# OpenAI 호환(vllm)을 위한 ChatOpenAI import
from langchain_openai import ChatOpenAI
# chat template 적용을 위한 ChatPromptTemplate import
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
# openai 툴 에이전트 및 실행을 위한 create_openai_tools_agent, AgentExecutor import
from langchain.agents import AgentExecutor

# -------------------------- tool 함수에 적용할 LLM vLLM에 연동하기 위한 LLM 래퍼 설정 --------------------------
# vLLM의 OpenAI 호환 서버에 LLM을 연동하기 위한 LLM 래퍼를 생성
tool_use_llm = ChatOpenAI(
                    # 모델 경로 설정
                    model= "/home/addinedu/LLM/LLM_Model",
                    # local 주소 설정
                    base_url= "http://localhost:8000/v1",
                    # api key는 사용 안함
                    api_key="EMPTY",
                    # llm의 일관성을 높히기 위해 temperature는 낮게 설정
                    temperature=0.3)

# --------------------------vLLM에 연동하기 위한 LLM 래퍼 생성 --------------------------
# vLLM의 OpenAI 호환 서버에 LLM을 연동하기 위한 LLM 래퍼를 생성
main_llm = ChatOpenAI(
                    # 모델 경로 설정
                    model= "/home/addinedu/LLM/LLM_Model",
                    # local 주소 설정
                    base_url= "http://localhost:8000/v1",
                    # api key는 사용 안함
                    api_key="EMPTY",
                    # llm의 일관성을 높히기 위해 temperature는 낮게 설정
                    temperature=0.3,
                    # vLLM에 툴 자동 선택을 활성화
                    model_kwargs={"tool_choice": "auto"})

# --------- tool 함수에 적용할 LLM에 응답하면 답변만 반환하는 함수 선언 ---------------------
def tool_llm_response(prompt):
    # tool_use_llm에 prompt를 입력하면 응답을 tool_use_llm에 입력
    tool_llm_answer = tool_use_llm.invoke(prompt)
    # tool_llm_answer가 str 형식이면 바로 tool_llm_response에 저장하고 메시지 형태이면 "content"를 저장하고 없으면 ""를 저장
    tool_llm_response = tool_llm_answer if isinstance(tool_llm_answer, str) else getattr(tool_llm_answer, "content", "") 
    # llm답변만 공백 제거후 return
    return (tool_llm_response or "").strip()
# -------------------------- tool 함수 선언 --------------------------
@tool("item_info",description= "상품/대상의 재고 관련된 질문이 입력되면 실행", return_direct=True)
# 상품 정보를 요청하면 mockup 재고 데이터로 응답하는 간단한 item_info 함수 선언
def item_info(text):
    # mockup 답변 설정
    mockup_answer = "재고는 2개 입니다."
    # mockup 답변 return
    return mockup_answer

@tool("item_coordinate",description= "상품/대상의 위치 관련된 질문이 입력되면 실행", return_direct=True)
# 상품 위치를 요청하면 mockup 좌표를 응답하는 간단한 item_coordinate 함수 선언
def item_coordinate(text):
    # mockup 답변 설정
    mockup_answer = "상품 위치는 x:100,y:21,z:0 입니다."
    # mockup 답변 return
    return mockup_answer

@tool("move_place", description="장소를 목적지로 이동을 요청하였을 때 입력된 문장에서 목적지(장소명) 1개만 추출", return_direct=True)
# 장소를 기반으로 한 장소 이동 요청을 하였을 때 LLM을 사용하여 파싱하여 return할 함수 선언 
def move_place(text):
    # llm에 입력할 프롬프트 규칙을 알려주는 지시문 설정
    prompt = (
        "역할: 한국어 장소 이동 명령에서 목적지 장소명 1개만 추출 \n"
        "규칙:\n"
        "1) 결과는 명사/명사구 1개(조사/따옴표/기호/마침표 제거)\n"
        "2) '~으로/~로/~까지' 등 조사는 삭제.\n"
        "3) 장소가 없거나 애매하면 '없음'만 출력.\n"
        "4) 응답은 딱 한 줄, 장소명만 출력.\n"
        "예시: '포장대로 가줘'→포장대, '신선식품 코너로 이동'→신선식품, '과자로 가줘'→과자\n"
        f"\n사용자 입력: {text}\n응답:"
    )
    # prompt를 입력하여 llm에서 응답 받기
    llm_response = tool_llm_response(prompt)
    return llm_response


# 정의한 Tool 함수를 에이전트가 사용할 수 있도록 리스트로 생성
tools = [item_info, item_coordinate, move_place]

# -------------------------- 시스템 규칙 및 툴 에이전트 생성--------------------------
# 프롬프트 규칙을 설정
# OpenAI 규칙에 따라서 system,human을 적용
# system : LLM에 알려줄 시스템 규칙 명시
# hunam : input에 사용자 입력이 입력
prompt = ChatPromptTemplate.from_messages([
    # system으로 모델에게 시스템에서 적용하야할 지시문을 작성
    ("system",
     "한국어만 사용.\n"
     "도구 사용 규칙:\n"
     "- 재고/수량 문의 → item_info\n"
     "- 상품 위치/좌표 문의 → item_coordinate\n"
     "- '~로/~으로/~까지' 등 목적지가 포함된 장소 이동 → move_place\n"
     "- 거리/방향(앞/뒤/전진/후진, m/cm 등) 포함 → move_distance\n"
     "도구를 사용했다면, 도구가 반환한 문자열을 그대로 사용자에게 출력\n"
     "도구가 불필요하면 간단명료하게 답하라."
    ),  
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")])

# OpenAI 툴 규약을 따르는 LangChain 에이전트 생성
agent = create_tool_calling_agent(main_llm,tools,prompt)
# 생성된 에이전트를 실행가능하도록 랩핑
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 메인 함수 선언
def main():
    # 무한 반복 
    while True:
        # 사용자 입력
        user_input = input("\n[LLM_message] :").strip()
        if user_input =='종료':
            break
        # 에이전트에 입력을 전달하고 실행
        llm_response = executor.invoke({"input": user_input})
        llm_out = llm_response["output"]
        # 디버그 print
        print("--- LLM 답변 ---")
        print(f"[LLM] {llm_out}")

# 메인 함수 실행
if __name__ =="__main__":
    main()