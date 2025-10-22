# ----------------------- STT 관련 라이브러리 선언 -----------------------
# 시스템 핸들을 위헤 sys 라이브러리 import
import sys
# 시간 handle을 위해 time 라이브러리 import
import time
# 오디오 값 저장을 위해 queue 라이브러리 import
import queue
# 배열 handle을 위해 numpy 라이브러리 import
import numpy as np
# 마이크 입력/스피커 출력 등 오디오 장치 제어를 위해 sounddevice 라이브러리 import
import sounddevice as sd
# gpu 사용 및 torch 연산을 위해 torch 라이브러리 import
import torch
# STT를 위해 whisper 라이브러리 import
import whisper


# 마이크 관련 class 선언
class Mike_Util():
    def __init__(self):
        # 마이크 모델 이름 설정
        self.mike_model_name = "K66"

    # 마이크 장치 목록 조회하여 튜플 리스트로 반환하는 함수 선언
    def mike_list_return(self):
        # sound_device로 연결된 하드웨어 장치 불러오기
        apis = sd.query_hostapis()
        # alsa 장치만 불러오기
        # alsa = advanced Linux Sound Architecture -> 리눅스에서 오디오 장치를 직접 제어하는 커널 레벨 사운드 시스템
        alsa_api = next((i for i,a in enumerate(apis) if 'alsa' in a['name'].lower()), None)
        # 시스템에 연결된 모든 오디오 장치 정보 sounddevie로 불러와서 devices에 저장
        devices = sd.query_devices()
        # 마이크 장치만 담을 list 선언
        mike_lists = []
        # device에 저장된 값을 i = 인덱스 번호, d = 저장된 값으로 구분
        # enumerate = 리스트 같은 시쿼스를 순회하면서 (index, 값)으로 반환하는 내장 함수
        for i,d in enumerate(devices):
            # max_input_channels가 0보다 크면 사운드 입력(마이크) 장치로 반단하여 mike_lists에 저장
            # 저장된 값에서 max_input_channels가 0보다 크면
            # --> max_input_channels가 마이크로 활용 가능
            # get : d에 딕셔너리가 저장되어 있음, get은 확인할 (key,해당 key가 없을 때 반환값)으로 구성됨
            if d['hostapi'] == alsa_api and d['max_input_channels'] > 0:
                # 마이크 장치만들 담을 list에 인덱스 번호와 이름을 저장
                mike_lists.append((i,d["name"]))
        # 마이크 장치 리스트 return
        return mike_lists
    
    # 마이크 리스트에서 원하는 마이크 인덱스를 반환하는 함수 선언
    def pick_mike_index(self,mike_info):
        # mike_info의 index, name 중
        for idx, name in mike_info:
            # K66이 이름에 있거나 USB_Audio가 있으면
            if(self.mike_model_name in name) or ("USB_Audio" in name) or ("USB Audio" in name):
                # 그 인덱스를 반환
                return idx     
        # mike_info의 index, name 중
        for idx, name in mike_info:
            # 장치명이 pulse인 입력 장치가 있으면 (K66, USB_Audio가 없으면)
            if name.strip().lower() == "pulse":
                # 인덱스 반환
                return idx                
        # mike_info의 index, name 중
        for idx, name in mike_info:
            if name.strip().lower()== "default":
                # 장치명이 default인 장치가 있으면
                return idx       
        # 위 조건이 전부 없다면 mike_info의 첫번째 인덱스를 반환하거나 None을 반환
        return mike_info[0][0] if mike_info else None
    
# STT 관련 STT_Util 선언
class STT_Util():
    def __init__(self):
        # 오디오 입력에서 한번에 전달받는 오디오 큐 크기 설정
        self.block_size = 1024
        # 한번에 입력받을 음성 길이 설정 [ms] 단위
        self.block_ms = 32
        # 음성 입력 전 주변 소음을 측정하여 에너지 기준을 잡기위한 초 설정
        self.noise_sec = 1
        # 발화 시작 판단 임계 배수(주변 소음 평균 배수 초과로 power가 감지되면 '말 시작'으로 간주)
        self.start_factor = 15
        # 발화 종료 판단 임계 배수(주변 소음 평균 배수 미만으로 에너지가 감지되면 '말 종료'로 간주)
        self.end_factor = 1.4
        # 말 종료가 이 시간 이후로 지속되면 음성 입력을 종료
        self.end_sec = 1.0
        # 음성 입력 최대 초 설정(30초)
        self.max_speak_sec = 7
        # 사용 언어 설정
        self.language = "ko"
        # 음성 인식을 저장할 큐 객체 선언
        self.audio_queue = queue.Queue()

    # sample_rate 기반 블록 크기 계산
    def compute_block_size(self, sample_rate):
        # 샘플 레이트에 현재 block_ms를 곱하여 콜백 1회당 처리할 블록 수를 계산
        bs = int(sample_rate * (self.block_ms / 1000.0))
        # 샘플 수를 최소 256, 최대 4096 범위로 설정
        return max(256, min(4096, bs))

    # 입력된 음성 입력의 power를 계산하는 함수
    def calculate_power(self,audio_input):
        # 입력된 음성이 빈 배열의 경우 
        if audio_input is None or getattr(audio_input, "size", 0) == 0:
            # 0.0을 return
            return 0.0
        # 음성 신호의 Power 값 연산
        power = (float(np.sqrt(np.mean(audio_input.astype(np.float32)**2))+1e-12))**2
        # 입력된 음성의 RMS 값을 return
        return power
    
    # 일정 시간 동안 입력된 음성에서 노이즈를 연산하는 함수 선언
    def calculate_noise(self): 
        # 입력된 음성의 순간 크기를 담을 list 선언
        input_sound_energies = []
        # 계산 시작 시간을 저장
        start_time = time.time()
        # 주어진 보정 시간 동안 반복
        while time.time()- start_time < self.noise_sec:
            # 에러가 없으면
            try:
                #설정한 계산 시간 이내에 q에서 오디오 값을 받아오기
                block = self.audio_queue.get(timeout = self.noise_sec)
            # 큐가 비어있으면
            except queue.Empty:
                # 다음 루프로 넘어가기
                continue
            # 큐에 저장한 블록의 power 값을 계산하여 input_sound_energies에 저장
            input_sound_energies.append(self.calculate_power(block))
        # 수집된 값이 있으면 평균 값을 반환, 없으면 아주 작은 값을 반환
        return float(np.mean(input_sound_energies)) if input_sound_energies else 1e-3
    
    # 발화 시작/종료를 감지하여 입력한 음성 오디오만 하나의 파일로 묶어서 반환하는 함수 선언
    def extract_input_audio(self):
        # 발화 시작 플래그 선언
        speech_start_flag = False
        # 마지막으로 목소리가 감지된 시간 저장
        last_speech_time = None
        # 음성인식 값을 담아둘 버퍼 리스트 선언
        speech_buffer = []
        # 한번 발화된 음성의 인식 시작 시간을 기록
        speech_start_time = time.time()
        # 1초동안 인식한 주변 소음의 크기
        input_sound_energies = self.calculate_noise()
        # 발화 시작을 주입된 음성의 크기에 시작 임계값을 곱한값을 기준으로 연산
        start_threshold = input_sound_energies * self.start_factor
        # 발화 종료를 주입된 음성의 크기에 종료 임계값을 곱한값을 기준으로 연산
        end_threshold = input_sound_energies * self.end_factor
        # 발화 종료가 될 때 까지
        while True:
            # 최대 음성 인식 시간을 초과하면
            if time.time() - speech_start_time > self.max_speak_sec:
                # 루프를 종료
                break
            # 에러가 없으면
            try:
                # 타임아웃 이내에 음성 인식 큐에서 오디오 블록을 가져옴
                block = self.audio_queue.get(timeout = 1.0)
            # 큐가 비어있으면
            except queue.Empty:
                # 다음 반복으로 넘어감
                continue
            # 현재 음성 인식 블록의 power값을 연산
            current_audio_energy = self.calculate_power(block)
            # 아직 발화가 시작되지 않았다면
            if not speech_start_flag:
                # 현재 음성 인식 블록의 rms값이 시작 임계값을 넘으면 
                if current_audio_energy > start_threshold:
                    # 발화 시작으로 판정
                    speech_start_flag = True
                    # 마지막으로 목소리가 감지된 시간을 현재 시간으로 설정
                    last_speech_time = time.time()
                    # 현재 음성 인식 블록을 음성 인식 버퍼에 저장
                    speech_buffer.append(block)
            # 이미 발화가 시작된 상태라면
            else:
                # 현재 음성 인식 블록을 계속 음성인식버퍼에 저장
                speech_buffer.append(block)
                # 현재 음성 인식 블록의 rms값이 종료 임계값보다 크면
                if current_audio_energy > end_threshold:
                    # 아직 말하고 있다고 인식하여 마직막 목소리가 감지된 시간 현재 시간으로 설정
                    last_speech_time = time.time()
                # 현재 음성 인식 블록의 rms값이 종료 임계값보다 작고
                else :
                    # 마지막으로 발화한 시간이 설정한 종료 시간 이상인 경우
                    if (time.time()- last_speech_time) >= self.end_sec:
                        # 루프 종료
                        break
        # 만약 음성 인식 버퍼가 존재한다면
        if speech_buffer:
            # numpy 배열로 이어붙여서 반환
            return np.concatenate(speech_buffer, axis=0).astype(np.float32)
        # 아무것도 음성 인식 버퍼에 저장되지 않았다면
        else:
            # 빈 배열을 반환
            return np.empty((0,), dtype=np.float32)
        
    # sounddevice.InputStream : 마이크 입력장치에서 오디오 데이터를 실시간으로 받아오기 위한 객체
    # sounddevice.InputStream이 실행될 때 마다 callback 될 함수 선언
    def audio_callback(self, indata, frames, time_info, status):
        # 콜백 중 경고/에러 상태가 있으면 표준 에러로 출력
        if status:
            # 현재 상태 표준 에러로 print
            print(status, file=sys.stderr)
        # 오디오 큐에 음성인식 데이터 입력
        self.audio_queue.put(indata.copy().reshape(-1))


    # 샘플 레이트에 따라 오디오를 16k로 리샘플링(선형보간) 해주는 함수 선언
    def resample_linear(self,audio, orig_sr, target_sr):
        # 기존 샘플레이트와 타켓 샘플레이크가 같거나 입력이 없으면
        if orig_sr == target_sr or audio.size == 0:
            # 기존 오디오를 그대로 반환
            return audio.astype(np.float32, copy=False)
        # 현재 음성 길이에 샘플레이트 변환 스케일을 곱하여 샘플 길이 연산
        out_len = int(round(audio.shape[0] * target_sr / float(orig_sr)))
        # 산출 길이가 1이하이면
        if out_len <= 1:
            # 빈 오디오 배열 반환
            return np.zeros((1,), dtype=np.float32)
        # 기존 오디오의 인덱스 축 생성
        old_index = np.linspace(0, audio.shape[0] - 1, num=audio.shape[0], dtype=np.float64)
        # 새 샘플길이에 따라 인덱스 축 생성 
        new_index = np.linspace(0, audio.shape[0] - 1, num=out_len, dtype=np.float64)
        # 원래 음성의 인덱스와 신규 인덱스에 따라 오디오를 보간
        new_audio = np.interp(new_index, old_index, audio.astype(np.float64))
        return new_audio.astype(np.float32)
        
    # stt 응답을 얻어와서 반환하는 함수 선언
    def stt_response(self,target_idx,stt_model,smaple_rate,fp16):
        # 입력 음성에 맞춰 블록 크기 연산
        block_size = self.compute_block_size(smaple_rate)
        # 입력 스트림을 설정하고, 블록 단뒤로 audio_callback이 호출되도록 설정
        with sd.InputStream(
                   callback=self.audio_callback,
                    blocksize=block_size,  # [CHG] 고정값(1024) → 동적(block_size)
                    samplerate=smaple_rate,
                    channels=1,
                    device=target_idx,
                    dtype="float32",):
            # 사용자에게 음성인식 안내
            print("---------------- Please Talk to me ---------------")
            # 사용자 음성 추출하여 audio에 저장
            audio = self.extract_input_audio()
            # Whisper는 16kHz를 기대하므로, 48k/44.1k를 지원하는 마이크 샘플 레이트도 16k로 변환
            audio_16k = self.resample_linear(audio, smaple_rate, target_sr=16000)
            # 사용자 음성을 whisper STT에 적용
            result = stt_model.transcribe(audio_16k,language=self.language, fp16=fp16)
            # 인식 결과 딕셔너리에서 텍스트만 추출하고 앞뒤 공백 제거
            stt_text = (result.get("text") or "").strip()
            # 결과 print
            # print(f"[음성 인식 결과] : {stt_text}")
            return stt_text