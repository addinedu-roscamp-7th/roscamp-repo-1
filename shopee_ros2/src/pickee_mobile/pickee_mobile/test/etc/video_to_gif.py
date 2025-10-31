#!/usr/bin/env python3
"""
WebM to GIF Converter
배속 2배, FPS 5, 압축률 50%, 원본 전체 구간 변환
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def get_video_duration(input_file):
    """비디오 파일의 총 재생 시간을 초 단위로 반환"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError:
        print(f"Error: 비디오 파일 '{input_file}'의 정보를 가져올 수 없습니다.")
        return None
    except ValueError:
        print(f"Error: 비디오 파일 '{input_file}'의 재생 시간을 파싱할 수 없습니다.")
        return None

def webm_to_gif(input_file, output_file=None, speed=3.0, fps=3, quality=20):
    """
    WebM 파일을 GIF로 변환
    
    Args:
        input_file (str): 입력 WebM 파일 경로
        output_file (str): 출력 GIF 파일 경로 (None이면 자동 생성)
        speed (float): 배속 (2.0 = 2배속)
        fps (int): 출력 GIF의 FPS
        quality (int): 압축률 (1-100, 낮을수록 높은 압축률)
    """
    
    # 입력 파일 확인
    if not os.path.exists(input_file):
        print(f"Error: 입력 파일 '{input_file}'을 찾을 수 없습니다.")
        return False
    
    # 출력 파일명 자동 생성
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.gif')
    
    # 비디오 전체 재생 시간 가져오기
    duration = get_video_duration(input_file)
    if duration is None:
        return False
    
    print(f"입력 파일: {input_file}")
    print(f"출력 파일: {output_file}")
    print(f"원본 재생 시간: {duration:.2f}초")
    print(f"변환 설정: {speed}배속, {fps}fps, 품질 {quality}%")
    
    try:
        # FFmpeg 명령어 구성
        # 1. 전체 구간 (0초부터 끝까지)
        # 2. 2배속 (setpts=0.5*PTS로 재생 속도 2배 빠르게)
        # 3. FPS 5로 설정
        # 4. 압축률 향상 (색상 수 제한 및 최적화)
        
        # 품질에 따른 색상 수 계산 (quality 50% = 128색상)
        max_colors = max(16, min(256, int(256 * quality / 100)))
        
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', '0',  # 시작 시간: 0초
            '-t', str(duration),  # 재생 시간: 전체
            '-vf', f'setpts={1/speed}*PTS,fps={fps},scale=iw*0.8:ih*0.8:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors={max_colors}:reserve_transparent=0[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
            '-loop', '0',  # 무한 반복
            '-y',  # 기존 파일 덮어쓰기
            str(output_file)
        ]
        
        print("변환 중...")
        print(f"실행 명령: {' '.join(cmd)}")
        
        # FFmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"✅ 변환 완료: {output_file}")
        
        # 파일 크기 정보 출력
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"원본 크기: {input_size:.2f} MB")
        print(f"변환 후 크기: {output_size:.2f} MB")
        print(f"압축률: {((input_size - output_size) / input_size * 100):.1f}%")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: FFmpeg 실행 중 오류가 발생했습니다.")
        print(f"명령어: {' '.join(cmd)}")
        print(f"오류 메시지: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: 예상치 못한 오류가 발생했습니다: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='WebM 파일을 GIF로 변환합니다')
    parser.add_argument('input', help='입력 WebM 파일 경로')
    parser.add_argument('-o', '--output', help='출력 GIF 파일 경로 (생략시 자동 생성)')
    parser.add_argument('-s', '--speed', type=float, default=2.0, help='배속 (기본값: 2.0)')
    parser.add_argument('-f', '--fps', type=int, default=3, help='출력 FPS (기본값: 5)')
    parser.add_argument('-q', '--quality', type=int, default=50, help='품질 1-100 (기본값: 50)')
    
    args = parser.parse_args()
    
    # FFmpeg 설치 확인
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg가 설치되어 있지 않습니다.")
        print("설치 방법: sudo apt install ffmpeg")
        return 1
    
    # 변환 실행
    success = webm_to_gif(
        input_file=args.input,
        output_file=args.output,
        speed=args.speed,
        fps=args.fps,
        quality=args.quality
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())