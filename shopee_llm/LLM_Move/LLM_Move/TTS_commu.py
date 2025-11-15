#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------ 범용 라이브러리 import -----------------
# 파일 제어를 위한 os 라이브러리 import
import os 
# 시스템 핸들을 위한 sys 라이브러리 import
import sys
# 입출력을 위한 io 라이브러리 import
import io
# 비동기 코드 실행을 위한 asyncio 라이브러리 import
import asyncio
# 함수형 문맥 관리자 contextmanager import -> 경고 제거를 위해 사용
from contextlib import contextmanager
# ------------------ TTS 관련 라이브러리 import -------------------
# edge_tts 활용을 위한 edge_tts 라이브러리 import
import edge_tts
# python에서 오디오 핸들을 위해 AudioSegment 라이브러리 import
from pydub import AudioSegment
# AudioSegment 객체 재생을 위해 play 라이브러리 import 
from pydub.playback import play

# TTS를 실행시켜 주는 TTS_Util 클래스 선언
class TTS_Util:
    def __init__(self):
        # 사용할 음성 voice 설정
        self.voice = "ko-KR-SoonBokNeural"
        # TTS 말하기 속도 선언
        self.rate = "+0%"
        # TTS 볼륨값 선언
        self.volume = "+100%"
    # tts 함수 설정
    async def speak(self, text):
        # edge_tts에 text와 voice를 적용
        tts = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)
        # TTS로 생성된 오디오 데이터를 저장할 빈 바이트형 변수 선언
        audio_bytes = b""
        # 비동기 스트림을 통해 tts 데이터에서 chunk를 순차적으로 받아오기
        async for chunk in tts.stream():
            # chunk 타입이 오디오면
            if chunk["type"] == "audio":
                # 오디오 데이터에 저장
                audio_bytes += chunk["data"]
         # audio 데이터를 mp3 형식으로 읽기
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        # mp3로 변환된 TTS 데이터 재생
        play(audio)