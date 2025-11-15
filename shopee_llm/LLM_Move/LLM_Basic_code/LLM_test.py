# ----------------------------- 기본 라이브러리 import -----------------------------
# GPU 사용을 위한 torch 라이브러리 import
import torch
# 토크나이저를 로드/저장하고 텍스트를 토큰 ID로 변환하는 데 사용하는 AutoTokenizer import
from transformers import AutoTokenizer
# 언어 모델을 로드/추론하는 AutoModelForCausalLM import
from transformers import AutoModelForCausalLM
# 4bit/8bit 양자화 설정을 위한 BitsAndBytesConfig import
from transformers import BitsAndBytesConfig
# ----------------------------- 로컬 모델 경로 ---------------------------------------
# 모델이 저장된 경로 설정
MODEL_PATH = "/home/addinedu/LLM/LLM_Model"  
# ----------------------------- 4bit 양자화 설정 -----------------------------
# 4bit 양자화를 위한 설정 객체 생성
quantization_4bit = BitsAndBytesConfig(
    # 모델 가중치를 4bit로 로드하도록 설정
    load_in_4bit=True,
    # 4bit로 연산 시 내부 계산에 사용할 dtype을 설정합니다
    q_4bit_compute_dtype=torch.bfloat16,
    # 4bit 양자화 방식으로 nf4(정규화된 4비트)를 사용
    q_4bit_type="nf4"
)
# ----------------------------- 토크나이저/모델 로드 (로컬 경로) -----------------------------
# 로컬 경로에서 tokenizer 로드
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
# # 로컬 경로에서 LLM 로드
model = AutoModelForCausalLM.from_pretrained(
    # 모델이 위치한 경로
    MODEL_PATH,
    # gpu가 있으면 자동으로 적용
    device_map="auto",
    # 4bit 양자화 사용
    quantization_config=quantization_4bit)

# ----------------------------- 한국어 대화 프롬프트 작성 -----------------------------
messages = [
    {"role": "system", "content": "반드시 한국어로만 대답해줘."},
    {"role": "user",   "content": "안녕! 간단히 자기소개 해줘."},
]

# ----------------------------- Qwen이 지원하는 chat template를 적용-----------------------------
prompt = tok.apply_chat_template(
    # 메시지 적용
    messages,
    # 생성 시작을 알리는 assistant 프롬프트 토큰을 자동으로 덧붙임
    add_generation_prompt=True,
    # 토큰화하지 않고 문자열 그대로 반환
    tokenize=False
)

# ----------------------------- 텐서 변환 및 생성 -----------------------------
# prompt 문자열을 텐서로 변환하고 모델이 위치한 gpu,cpu에 적용
inputs = tok(prompt, return_tensors="pt").to(model.device)
# LLM에 제작한 prompt 적용하여 답변을 저장
outputs = model.generate(
    # imput텐서를 unpack하여 전달
    **inputs,
    # 생성할 최대 토큰 수 설정
    max_new_tokens=120,
    # 샘플링 온도 설정(높을수록 창의적, 낮을 수록 결정적)
    temperature=0.7,
    # 누적 확률의 상위만 고려
    # 누적 확률 : 다음 토큰의 확률 분포를 고려하여 가장 가능성이 높은 후보만 적용
    top_p=0.9,
    # 무작위 샘플링을 적용
    do_sample=True
)

# ----------------------------- 결과 출력 -----------------------------
# 전체 답변(입력+생성)
full_token = outputs[0]
# 입력 프롬프트 길이 (여기까지는 프롬프트 토큰)
promt_token = inputs["input_ids"].shape[1]
# 전체 토큰에서 입력 프로프트까지 슬라이스(llm 답변만 남음)
llm_response_token = full_token[promt_token:]
# messages에서 유저 질문 가져오기
usr_prompt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
# llm_response_token 사람이 읽을 수 있는 문자열로 디코딩
llm_response = tok.decode(llm_response_token, skip_special_tokens=True)
print(f"[User] {usr_prompt}")
# 모델이 생성한 토큰을 사람이 읽을 수 있는 문자열로 디코딩합니다
print(f"[LLM] {llm_response}")