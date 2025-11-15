# 파일 경로를 다루기 위해 os라이브러리 import
import os
# gpu 연산을 다루기 위해 torch라이브러리 import
import torch
# 데이터 셋에서 Jsonl을 로드하기 위한 load_dataset 라이브러리 import
from datasets import load_dataset
# 경량.고속 LLM 호출을 위한 FastLanguageModel 라이브러리 import
from unsloth import FastLanguageModel
# Qwen 등 모델별 대화 템플릿(chat template)적용을 위해 get_chat_template 라이브러리 import
from unsloth.chat_templates import get_chat_template    
# 모델 train을 위한 SFTTrainer, SFTConfig 라이브러리 import
from trl import SFTTrainer, SFTConfig

# --------------------------- 모델 경로 및 데이터 셋 경로 설정 ---------------------
# tunning 할 base 모델 경로
BASE_MODEL_DIR = "/home/addinedu/LLM/LLM_Model"
#  파인튜닝 결과물을 저장할 경로
OUT_DIR = "/home/addinedu/LLM/LLM_Tuing_Model"
# 학습에 사용할 jsonl 데이터 경로
TRAIN_DATA = "/home/addinedu/LLM/Tunning_data/Qwen_place_tunning_data/llm_train_assistant.jsonl"

# --------------------------- 학습 설정 파라미터 설정 ---------------------
# 최대 토큰 길이 
MAX_LEN   = 512
# QLoRA 사용 여부(4bit로 베이스 모델 양자화 하여 학습)
LOAD_4BIT = True
# 연산 정밀도
DTYPE     = torch.bfloat16
# 학습률
LR        = 2e-4
# 전체 데이터셋을 몇 번 반복해서 학습할지 설정(에폭 수)
EPOCHS    = 2
# GPU당 할당 배치 크기
# batch : 한번의 모델 업데이트에 함께 넣어서 학습시키는 샘플(대화) 묶음
BATCH_SIZE    = 1
# 누적 스텝 수
# 한꺼번에 큰 batch를 입력하면 수렴이 안정적이지만 ram이 부족할 수 있음, 
# 여러 미치 배치를 누적했다가 정회진 횟수 마다 모델 학습에 적용
GRAD_ACCUM    = 16 
# --------------------------- 모델 및 토크나이저 로드 ---------------------
# Unsloth의 from_pretrained로 베이스 모델과 토크나이저를 로드
model,tokenizer = FastLanguageModel.from_pretrained(
    # 모델 경로는 사전 저장 경로로 설정
    model_name= BASE_MODEL_DIR,
    # 최대 입력 길이 설정
    max_seq_length= MAX_LEN,
    # 연산 정밀도 설정
    dtype= DTYPE,
    # QLoRa 설정
    load_in_4bit= LOAD_4BIT,
    # LoRA 기반 어뎁터 학습을 할 것이기 때문에 full_finetuning을 False로 설정
    full_finetuning= False)

# Qwen 2.5에 맞는 대화 탬플릿을 토크나이저에 부착
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# LoRA 어댑터를 모델에 부착
# PEFT(Parameter-Efficient Fine-Tuning) 
# -> 전체 모델 가중치를 다 업데이트하지 않고 일부만 학습하는 방식
model = FastLanguageModel.get_peft_model(
    # 대상 모델 설정
    model           = model,
    # r : LpRA 랭크(저랭크 차원), 추가로 학습하는 가중치를 결정하는 파라미터
    r               = 16,
    # 어뎁터를 부착할 계층 설정
    target_modules  = ["q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj"],
    # LoRA 스케일 계수 설정
    lora_alpha      = 32,
    # LoRA  경로에 적용되는 학습 중에 확률을 확 낮추어서 
    # 과적합(같은 데이터를 여려번 학습)을 줄이고 일반화하는 성능 지표
    lora_dropout    = 0.0,
    # 바이어스 학습 None으로 설정
    bias            = "none",
    # unsloth방식의 체크포인팅을 활용
    use_gradient_checkpointing = "unsloth"
)

# --------------------------- 학습 준비 ---------------------
# 데이터 파일 목록 구성
# train key로 불러오기
data_files = {"train": TRAIN_DATA}
# 학습 데이터 읽어오기
train_dataset = load_dataset("json",data_files=data_files)

# Qwen에서 요구하는 형식으로 학습 데이터 변환
def formatting_func(examples):
    # 데이터 셋에서 messages 값 받아오기
    convos = examples["messages"]
    # 각 줄에 대해서 qwen_chat_template 적용
    # 토큰화 False, 응답 프롬프트 해더 붙이기 False(이미 데이터 셋에서 붙여줌)
    texts = [tokenizer.apply_chat_template(convo,tokenize=False,add_generation_prompt=False)
        for convo in convos]
    # datasets.map이 요구하는 형식으로 반환
    return {"text": texts}

# Qwen 탬플릿을 적용하고, 원래 컬럼(messages 등)은 제거하여 메모리 절감
train_ds = train_dataset["train"].map(formatting_func, batched=True, remove_columns=train_dataset ["train"].column_names)

# SFT 학습 설정
training_args = SFTConfig(
    # 체크포인트/로그 저장 폴더 설정(LoRA 어댑터 저장 폴더)
    output_dir                      = os.path.join(OUT_DIR, "adapter"),
    # 학습 에폭 수 설정
    num_train_epochs                = EPOCHS,
    # GPU 당 배치 크기 설정
    per_device_train_batch_size     = BATCH_SIZE,
    # 누적 스텝 수 설정
    gradient_accumulation_steps     = GRAD_ACCUM,
    # 학습률 설정
    learning_rate                   = LR,
    # 스케쥴러 종류 설정
    lr_scheduler_type               = "cosine",
    # 위밍업 비율 설정
    warmup_ratio                    = 0.03,
    # 로그 출력 간격 설정
    logging_steps                   = 10,
    # 체크 포인트 저장 간격
    save_steps                      = 200,
    # 평가 실행 전략
    eval_strategy                   = "no",
    # GPU가 가능하면 bf16으로 연산
    bf16                            = torch.cuda.is_available(),
    # 최대 시퀀스 길이 설정
    max_seq_length                  = MAX_LEN,
    # 여러 짧은 샘플은 한 시퀀스로 할지(Flase로 설정)
    packing                         = False)

# 학습 트레이너 객체 생성
model_trainer = SFTTrainer(
    model               = model,
    tokenizer           = tokenizer,
    train_dataset       = train_ds,
    args                = training_args,
    dataset_text_field  = "text")

# 학습 수행
print("========= 학습 시작 ============")
model_trainer.train()

# --------------------------- 저장 ---------------------
# 결과 저장 폴더에 adapter 디렉터리가 없으면 생성
os.makedirs(os.path.join(OUT_DIR, "adapter"), exist_ok=True)
#  LoRA 어댑터 가중치만 저장
model.save_pretrained_merged(os.path.join(OUT_DIR, "adapter"), tokenizer, save_method="lora")
# LoRA 어댑터와 베이스 모델을 합쳐 완전한 16-bit 모델 저장
os.makedirs(os.path.join(OUT_DIR, "tunning_model"), exist_ok=True)
# vLLM/Transformers에서 곧바로 로드 가능한 병합 모델을 저장
model.save_pretrained_merged(os.path.join(OUT_DIR, "tunning_model"), tokenizer, save_method="merged_16bit")
# 완료 메시지를 출력
print("=== Done ===")
# LoRA 어댑터 저장 경로를 출력
print("LoRA adapter  :", os.path.join(OUT_DIR, "adapter"))
# 병합 모델 저장 경로를 출력
print("Merged 16-bit :", os.path.join(OUT_DIR, "tunning_model"))