#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
타인 이미지 수집 도구
부정 예제(Negative Samples) 데이터셋 구축용
"""

import cv2
import os
import time
from datetime import datetime


def main():
    # 설정
    output_dir = 'employee_data/train/images'
    camera_index = 2  # 카메라 인덱스
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 카메라 초기화
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'Error: 카메라 {camera_index}를 열 수 없습니다.')
        return
    
    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 기존 파일 개수 확인 (이어서 저장)
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('other_') and f.endswith('.jpg')]
    count = len(existing_files)
    
    print('=' * 60)
    print('타인 이미지 수집 도구')
    print('=' * 60)
    print(f'저장 경로: {output_dir}')
    print(f'기존 이미지: {count}개')
    print()
    print('조작법:')
    print('  SPACE   : 현재 프레임 저장')
    print('  A       : 자동 캡처 모드 (1초마다 자동 저장)')
    print('  S       : 자동 캡처 중지')
    print('  ESC/Q   : 종료')
    print('=' * 60)
    
    auto_capture = False
    last_capture_time = 0
    auto_interval = 1.0  # 자동 캡처 간격 (초)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: 프레임을 읽을 수 없습니다.')
            break
        
        # 정보 표시
        display_frame = frame.copy()
        info_text = f'Images: {count} | Mode: {"AUTO" if auto_capture else "MANUAL"}'
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if auto_capture:
            cv2.putText(display_frame, 'AUTO CAPTURING...', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Negative Sample Capture', display_frame)
        
        # 자동 캡처 모드
        current_time = time.time()
        if auto_capture and (current_time - last_capture_time) >= auto_interval:
            filename = f'{output_dir}/other_{count:04d}.jpg'
            cv2.imwrite(filename, frame)
            print(f'[AUTO] Saved: {filename}')
            count += 1
            last_capture_time = current_time
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE - 수동 캡처
            filename = f'{output_dir}/other_{count:04d}.jpg'
            cv2.imwrite(filename, frame)
            print(f'[MANUAL] Saved: {filename}')
            count += 1
            
        elif key == ord('a') or key == ord('A'):  # A - 자동 캡처 시작
            auto_capture = True
            last_capture_time = current_time
            print('자동 캡처 모드 시작')
            
        elif key == ord('s') or key == ord('S'):  # S - 자동 캡처 중지
            auto_capture = False
            print('자동 캡처 모드 중지')
            
        elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC/Q - 종료
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print('=' * 60)
    print(f'총 {count}개 이미지 수집 완료')
    print('=' * 60)
    print()
    print('다음 단계:')
    print('1. 수집된 이미지를 확인하세요 (타인만 포함되어야 함)')
    print('2. 라벨 파일(.txt)은 생성하지 마세요 (Background로 학습)')
    print('3. yolo_train.py를 실행하여 재학습하세요')


if __name__ == '__main__':
    main()
