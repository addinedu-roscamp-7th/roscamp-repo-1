# 정규표현식 패턴 매칭을 위해 re 모듈을 임포트합니다.
import re
# JSON 직렬화/역직렬화를 위해 json 모듈을 임포트합니다.
import json
# 텐서 연산 및 디바이스 관리(GPU/CPU)를 위해 torch를 임포트합니다.
import torch
# 토크나이저 로드 및 텍스트→토큰 변환을 위해 AutoTokenizer를 임포트합니다.
from transformers import AutoTokenizer
# 생성형 언어모델 로드를 위해 AutoModelForCausalLM을 임포트합니다.
from transformers import AutoModelForCausalLM
# 4/8비트 양자화 설정을 위해 BitsAndBytesConfig를 임포트합니다.
from transformers import BitsAndBytesConfig

# 로컬에 저장한 Qwen 모델 폴더 경로를 지정합니다.
MODEL_PATH = "/home/addinedu/LLM/LLM_Model"

# 4비트 양자화 로드 설정 객체를 생성합니다.
quantization_4bit = BitsAndBytesConfig(
    # 모델 가중치를 4비트로 로드하여 VRAM 사용량을 줄입니다.
    load_in_4bit=True,
    # 4비트 연산 시 내부 계산 dtype을 bfloat16으로 설정합니다(또는 fp16 가능).
    q_4bit_compute_dtype=torch.bfloat16,
    # 4비트 양자화 스킴으로 nf4를 사용합니다.
    q_4bit_type="nf4"
)
# 로컬 경로에서 토크나이저를 로드합니다(빠른 Rust 토크나이저).
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
# 로컬 경로에서 언어모델을 로드합니다(4비트 양자화 적용).
model = AutoModelForCausalLM.from_pretrained(
    # 모델 파일이 위치한 디렉터리를 지정합니다.
    MODEL_PATH,
    # GPU가 있으면 자동으로 디바이스를 매핑합니다.
    device_map="auto",
    # 앞서 정의한 4비트 양자화 구성을 적용합니다.
    quantization_config=quantization_4bit)

# 재고 조회 더미 툴 함수: 결과를 로그로 출력하고 JSON 형태로 반환합니다.
def place_info(text):
    # 콘솔에 간단한 안내 메시지를 출력합니다.
    print("재고가 2개 남아있습니다.")
    # 모델이 참조할 수 있도록 구조화된 결과를 반환합니다.
    return {"tool": "place_info", "query": text, "answer": "재고가 2개 남아있습니다."}

# 좌표 조회 더미 툴 함수: 결과를 로그로 출력하고 JSON 형태로 반환합니다.
def coordinate(text):
    # 콘솔에 좌표 예시를 출력합니다.
    print("(x : 100,y: 200,z: 100)")
    # 모델이 참조할 수 있도록 구조화된 결과를 반환합니다.
    return {"tool": "coordinate", "query": text, "answer": {"x": 100, "y": 200, "z": 100}}

# OpenAI 호환 툴 스키마 목록을 정의합니다.
tools = [
    # 첫 번째 툴(place_info)의 스키마를 정의합니다.
    {
        # 이 엔트리가 함수 타입 툴임을 지정합니다.
        "type": "function",
        # 함수 이름/설명/인자 스키마를 담습니다.
        "function": {
            # 모델이 호출할 함수명을 지정합니다(실제 파이썬 함수명과 일치 권장).
            "name": "place_info",
            # 툴의 목적을 한국어로 설명합니다.
            "description": "상품/대상의 재고 상태나 배치 정보를 조회합니다.",
            # 함수 인자 정의(JSON Schema)를 제공합니다.
            "parameters": {
                # 최상위 인자 구조가 객체임을 지정합니다.
                "type": "object",
                # 사용할 속성(인자)들을 정의합니다.
                "properties": {
                    # 단일 인자 text를 정의합니다.
                    "text": {
                        # text는 문자열 타입입니다.
                        "type": "string",
                        # 모델이 넣을 값의 의미/예시를 설명합니다.
                        "description": "조회할 대상 설명 또는 상품명. 예: '사과', '우유', '사과 재고 알려줘'.",
                        # 빈 문자열 입력을 방지합니다.
                        "minLength": 1,
                        # 과도하게 긴 입력을 제한합니다.
                        "maxLength": 100
                    }
                },
                # 필수 인자로 text를 요구합니다.
                "required": ["text"],
                # 정의되지 않은 추가 키를 금지합니다.
                "additionalProperties": False
            }
        }
    },
    # 두 번째 툴(coordinate)의 스키마를 정의합니다.
    {
        # 이 엔트리 또한 함수 타입 툴입니다.
        "type": "function",
        # 함수 메타데이터 블록입니다.
        "function": {
            # 호출할 함수명을 지정합니다.
            "name": "coordinate",
            # 툴의 역할을 설명합니다.
            "description": "요청한 대상/장소의 좌표 정보를 조회합니다.",
            # 함수 인자 정의(JSON Schema)를 제공합니다.
            "parameters": {
                # 최상위 인자 구조가 객체임을 지정합니다.
                "type": "object",
                # 사용할 속성(인자)들을 정의합니다.
                "properties": {
                    # 단일 인자 text를 정의합니다.
                    "text": {
                        # text는 문자열 타입입니다.
                        "type": "string",
                        # 모델이 넣을 값의 의미/예시를 설명합니다.
                        "description": "좌표가 필요한 대상이나 장소명. 예: '포장대', '반납대', 'A-3 매대'.",
                        # 빈 문자열 입력을 방지합니다.
                        "minLength": 1,
                        # 과도하게 긴 입력을 제한합니다.
                        "maxLength": 100
                    }
                },
                # 필수 인자로 text를 요구합니다.
                "required": ["text"],
                # 정의되지 않은 추가 키를 금지합니다.
                "additionalProperties": False
            }
        }
    }
]

# 모델 출력에서 <tool_call>{...}</tool_call> JSON을 추출할 정규식을 준비합니다.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# 모델 출력에서 툴콜 JSON을 파싱해 dict로 반환하는 유틸 함수입니다.
def parse_tool_call(text: str):
    """모델 출력에서 <tool_call>{...}</tool_call> JSON을 추출해 dict로 반환."""
    # 첫 번째 매칭을 찾습니다.
    m = TOOL_CALL_RE.search(text)
    # 매칭이 없으면 툴 호출이 없는 것으로 간주합니다.
    if not m:
        return None
    try:
        # 캡처 그룹의 JSON 문자열을 dict로 변환해 반환합니다.
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        # JSON 형식 오류 시 None을 반환합니다.
        return None

# 모델의 동작 규칙과 툴 호출 형식을 안내하는 시스템 메시지를 정의합니다.
SYSTEM_RULES = (
    "당신은 한국어만 사용하는 어시스턴트입니다.\n"
    "필요하다면 아래 형식으로 툴을 호출하세요:\n"
    "<tool_call>{\"name\": \"툴이름\", \"arguments\": { ... }}</tool_call>\n"
    "- 가능한 툴: place_info(text:str), coordinate(text:str)\n"
    "- 정말 툴이 필요할 때만 호출하고, 불필요하면 바로 답변하세요."
)

# 테스트용 사용자 입력을 정의합니다(원하는 문장으로 교체하며 테스트 가능).
user_text = "사과 재고 알려줘"

# 1차 생성에 사용할 대화 이력을 구성합니다(시스템 규칙 + 사용자 요청).
messages = [
    # 모델 규칙/툴 사용 지침이 담긴 시스템 메시지입니다.
    {"role": "system", "content": SYSTEM_RULES},
    # 실제 사용자 요청입니다.
    {"role": "user",   "content": user_text},
]

# 툴 스키마를 템플릿에 주입하여 함수콜 친화적 출력이 나오도록 합니다.
prompt = tok.apply_chat_template(
    # 대화 이력을 전달합니다.
    messages,
    # OpenAI 호환 tools 스키마를 함께 전달합니다.
    tools=tools,
    # 어시스턴트 응답 시작 토큰을 자동으로 붙입니다.
    add_generation_prompt=True,
    # 문자열 프롬프트가 필요하므로 토큰화는 하지 않습니다.
    tokenize=False,
)

# 프롬프트를 텐서로 변환하고 모델 디바이스(GPU/CPU)에 올립니다.
inputs = tok(prompt, return_tensors="pt").to(model.device)

# 함수콜 판단 정확도를 높이기 위해 낮은 temperature로 1차 생성을 수행합니다.
outputs = model.generate(
    # 입력 텐서를 언팩하여 전달합니다.
    **inputs,
    # 최대 생성 토큰 수를 제한합니다.
    max_new_tokens=256,
    # 결정성을 높여 함수콜 파라미터가 안정적으로 나오게 합니다.
    temperature=0.2,
    # 누클리어스 샘플링 임계값을 설정합니다.
    top_p=0.9,
    # 샘플링 기반 생성을 활성화합니다.
    do_sample=True,
)

# 전체 출력 시퀀스에서 입력 길이만큼을 제외해 생성된 부분만 추출합니다.
full_ids = outputs[0]
# 입력 프롬프트 길이를 구합니다.
inp_len = inputs["input_ids"].shape[1]
# 생성된 토큰을 사람이 읽을 수 있는 문자열로 디코딩합니다.
gen_text = tok.decode(full_ids[inp_len:], skip_special_tokens=True)

# 1차 모델 생성 결과를 확인 출력합니다.
print("=== 1차 모델 출력 ===")
# 1차 생성 텍스트를 출력합니다.
print(gen_text)

# 생성 결과에서 툴 호출 JSON을 추출합니다.
tool_call = parse_tool_call(gen_text)

# 툴 호출이 있는 경우에만 실행 및 2차 생성을 진행합니다.
if tool_call:
    # 호출된 툴 이름을 가져옵니다.
    name = tool_call.get("name")
    # 호출 인자(dict)를 가져오되 None 방지로 기본값을 둡니다.
    args = tool_call.get("arguments", {}) or {}

    # 모델 툴 이름 → 실제 파이썬 함수 매핑 테이블입니다.
    tool_map = {
        "place_info": place_info,
        "coordinate": coordinate,
    }

    # 지원하는 툴이면 실행하고, 아니면 에러 객체를 만듭니다.
    if name in tool_map:
        # 스키마상 단일 인자 'text'를 안전하게 문자열로 변환합니다.
        text_arg = str(args.get("text", "")).strip()
        # 해당 툴 함수를 실행하여 결과(JSON)를 받습니다.
        tool_result = tool_map[name](text_arg)
    else:
        # 알 수 없는 툴 이름인 경우 에러 정보를 구성합니다.
        tool_result = {"error": f"Unknown tool: {name}", "raw": tool_call}

    # 실행된 툴의 결과를 로그로 출력합니다.
    print("\n=== 실행된 툴 결과 ===")
    # 툴 결과(JSON)를 출력합니다.
    print(tool_result)

    # 툴 결과를 대화 이력에 role="tool" 메시지로 추가합니다.
    messages2 = messages + [
        # 모델이 활용할 수 있도록 JSON 문자열로 전달합니다.
        {"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)},
        # 최종 응답을 한국어로 간결하게 생성하도록 지시합니다.
        {"role": "system", "content": "위 tool 결과를 반영하여 한국어로 간결한 최종 답변만 출력하세요."}
    ]

    # 2차 생성용 프롬프트를 구성합니다(툴 결과 반영).
    prompt2 = tok.apply_chat_template(
        # 갱신된 대화 이력을 전달합니다.
        messages2,
        # 동일한 tools 스키마를 전달하여 일관성을 유지합니다.
        tools=tools,  # 일관성 유지를 위해 계속 전달
        # 어시스턴트 응답 시작 토큰을 자동으로 붙입니다.
        add_generation_prompt=True,
        # 문자열 프롬프트가 필요하므로 토큰화는 하지 않습니다.
        tokenize=False,
    )
    # 2차 프롬프트를 텐서로 변환하여 모델 디바이스로 보냅니다.
    inputs2 = tok(prompt2, return_tensors="pt").to(model.device)

    # 최종 자연어 답변 생성을 수행합니다(자연스러움 위해 temperature↑).
    outputs2 = model.generate(
        # 입력 텐서를 언팩하여 전달합니다.
        **inputs2,
        # 최대 생성 토큰 수를 제한합니다.
        max_new_tokens=200,
        # 문장 자연스러움을 위해 온도를 다소 높입니다.
        temperature=0.7,
        # 누클리어스 샘플링 임계값을 설정합니다.
        top_p=0.9,
        # 샘플링 기반 생성을 활성화합니다.
        do_sample=True,
    )

    # 2차 생성 결과에서 입력 길이를 제외한 생성 부분만 추출합니다.
    full_ids2 = outputs2[0]
    # 2차 입력 프롬프트 길이를 구합니다.
    inp_len2 = inputs2["input_ids"].shape[1]
    # 최종 한국어 답변 텍스트를 디코딩합니다.
    final_answer = tok.decode(full_ids2[inp_len2:], skip_special_tokens=True)

# 툴 호출이 없으면 1차 생성물이 최종 답변이 됩니다.
else:
    # 1차 생성 텍스트를 최종 답변으로 사용합니다.
    final_answer = gen_text

# 구분선과 함께 최종 답변을 출력합니다.
print("\n=== 최종 한국어 답변 ===")
# 최종 한국어 답변을 출력합니다.
print(final_answer)
