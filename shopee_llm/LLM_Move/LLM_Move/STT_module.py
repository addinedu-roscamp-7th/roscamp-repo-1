# ----------------------- 커스텀 클래스 import ----------------------------
# 마이크 설정을 위한 Mike_Utile 클래스 import
from STT_commu import Mike_Util
# STT 관련 커스텀 함수 사용을 위한 STT_Util 클래스 import
from STT_commu import STT_Util
# -----------------------  STT 관련 라이브러리import ----------------------
# gpu 사용 및 torch 연산을 위해 torch 라이브러리 import
import torch
# STT를 위해 whisper 라이브러리 import
import whisper
# 마이크 입력/스피커 출력 등 오디오 장치 제어를 위해 sounddevice 라이브러리 import
import sounddevice as sd
# ----------------------- 기타 라이브러리 import -------------------------------
import sys

class STT_Module():
    def __init__(self):
        # Whisper에서 사용할 모델 선정
        # tiny/base/small/medium/large -> 클수록 정확도 up, 반응 down
        self.stt_model_name = "small"
        # STT smaple_rate 설정
        self.stt_sample_rate = 44100
        self.force_device = "cpu"
        # Mike_Util class 객체 선언
        self.mike_handle = Mike_Util()
        # STT_Util class 객체 선언
        self.stt_handle = STT_Util()
    def stt_use(self):
        # ----------------------- STT 설정 및 모델 세팅 -----------------------
        # 마이크 장치 목록 mike_input에 저장
        mike_input = self.mike_handle.mike_list_return()
        # 마이크 입력이 없으면 
        if not mike_input:
            print("사용가능한 마이크 입력 장치가 없습니다.")
            # 프로그램 종료
            sys.exit(1)
            # 마이크 장치 목록 print
        print("---------------- mic list ------------------")
        for idx,name in mike_input:
            print(f"[{idx}] {name}")
        # 마이크 장치 목록 중 targex_idx 설정
        target_idx = self.mike_handle.pick_mike_index(mike_input)
        # target_idx가 없으면 디버깅 문구 print
        if target_idx is None:
            print("입력 장치를 선택할 수 없습니다.")
            # 프로그램 종료
            sys.exit(1)
        # 선택된 장치 인덱스와 이름 출력
        print(f"사용할 마이크 index = {target_idx} {sd.query_devices(target_idx)['name']}")           
        # GPU 사용 가능하면 gpu로 사용, 사용 못하면 cpu를 사용
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.force_device
        # whisper 모델을 지정한 device에 로드
        model = whisper.load_model(self.stt_model_name, device=device)
        # GPU 사용가능하면 모델을 16bit로 양자화에 사용
        use_fp16 = torch.cuda.is_available()
        # sounddevice 기본 샘플 레이트를 16kHZ로 설정
        sd.default.samplerate = self.stt_sample_rate
        # 입력 채널 수를 1채널로 설정
        sd.default.channels =1
        # 입력 디바이스를 설정 (출력은 None)
        sd.default.device = (target_idx, None)
        # 사용자 음성으로 stt를 수행
        stt_result = self.stt_handle.stt_response(target_idx,model,self.stt_sample_rate,use_fp16)
        # STT 결과 print
        # print("[STT]", stt_result)
        return stt_result