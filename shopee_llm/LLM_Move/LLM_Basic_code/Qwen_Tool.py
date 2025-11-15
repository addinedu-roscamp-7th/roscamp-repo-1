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

# 정의한 Tool 함수를 에이전트가 사용할 수 있도록 리스트로 생성
tools = [item_info, item_coordinate]
# --------------------------vLLM에 연동하기 위한 LLM 래퍼 생성 --------------------------
# vLLM의 OpenAI 호환 서버에 LLM을 연동하기 위한 LLM 래퍼를 생성
vllm = ChatOpenAI(
    # 모델 이름 설정 (서버에서 이미 경로로 설정해줬기 때문에 식별용으로 작성만 하면 됨)
    model= "/home/addinedu/LLM/LLM_Model",
    # local 주소 설정
    base_url= "http://localhost:8000/v1",
    # api key는 사용 안함
    api_key="EMPTY",
    # llm의 일관성을 높히기 위해 temperature는 낮게 설정
    temperature=0.3,
    # vLLM에 툴 자동 선택을 활성화
    model_kwargs={"tool_choice": "auto"}
)
# -------------------------- 시스템 규칙 및 툴 에이전트 생성--------------------------
# 프롬프트 규칙을 설정
# OpenAI 규칙에 따라서 system,human을 적용
# system : LLM에 알려줄 시스템 규칙 명시
# hunam : input에 사용자 입력이 입력
prompt = ChatPromptTemplate.from_messages([
    # system으로 모델에게 시스템에서 적용하야할 지시문을 작성
    ("system", "한국어만 사용, 필요할 때만 도구(item_info, item_coordinate)를 호출,도구가 활용될 때는 도구의 answer를 그대로 답변해줘"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# OpenAI 툴 규약을 따르는 LangChain 에이전트 생성
agent = create_tool_calling_agent(vllm,tools,prompt)
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

