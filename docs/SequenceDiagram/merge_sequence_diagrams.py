#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시퀀스 다이어그램 파일들을 자동으로 병합하는 스크립트
PlantUML 형식의 파일들을 하나의 문서로 합칩니다.
"""

import os
import re
import glob
from pathlib import Path

def get_file_order():
    """파일들을 올바른 순서로 정렬합니다."""
    files = []
    
    for file_path in glob.glob("SC_*.md"):
        # SC_01_1.md, SC_02_2_1.md, SC_02_2_2.md 등의 패턴 처리
        if file_path == "SC_04.md":
            files.append((4, 0, 0, file_path))
        else:
            # SC_숫자_숫자_숫자.md 또는 SC_숫자_숫자.md 패턴
            match = re.match(r'SC_(\d+)_(\d+)(?:_(\d+))?\.md', file_path)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))
                sub_minor = int(match.group(3)) if match.group(3) else 0
                files.append((major, minor, sub_minor, file_path))
    
    # 정렬: (major, minor, sub_minor) 순서로
    files.sort(key=lambda x: (x[0], x[1], x[2]))
    return [file_path for _, _, _, file_path in files]

def clean_plantuml_content(content):
    """PlantUML 내용을 정리합니다."""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # @startuml과 @enduml 라인은 제거 (병합 시 중복 방지)
        if line.strip().startswith('@startuml') or line.strip().startswith('@enduml'):
            continue
        # 빈 라인은 유지
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def merge_sequence_diagrams():
    """시퀀스 다이어그램 파일들을 병합합니다."""
    print("시퀀스 다이어그램 병합을 시작합니다...")
    
    # 현재 디렉토리에서 SC_로 시작하는 파일들 찾기
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 파일 순서 정렬
    ordered_files = get_file_order()
    
    if not ordered_files:
        print("SC_로 시작하는 MD 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 파일들: {ordered_files}")
    
    # 병합된 내용 저장
    merged_content = []
    merged_content.append("@startuml 병합된_시퀀스_다이어그램")
    merged_content.append("!theme plain")
    merged_content.append("")
    
    # 각 파일 처리
    for i, file_path in enumerate(ordered_files):
        print(f"처리 중: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 파일명에서 제목 추출
            title = file_path.replace('SC_', '').replace('.md', '')
            
            # 섹션 구분자 추가
            merged_content.append(f"== {title} ==")
            merged_content.append("")
            
            # PlantUML 내용 정리하여 추가
            cleaned_content = clean_plantuml_content(content)
            merged_content.append(cleaned_content)
            
            # 파일 간 구분을 위한 빈 라인 추가
            if i < len(ordered_files) - 1:
                merged_content.append("")
                merged_content.append("---")
                merged_content.append("")
            
        except Exception as e:
            print(f"파일 {file_path} 처리 중 오류: {e}")
            continue
    
    # 마지막 @enduml 추가
    merged_content.append("")
    merged_content.append("@enduml")
    
    # 병합된 내용을 파일로 저장
    output_file = "merged_sequence_diagrams.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merged_content))
    
    print(f"\n병합 완료! 결과 파일: {output_file}")
    print(f"총 {len(ordered_files)}개 파일이 병합되었습니다.")

def create_html_output():
    """PlantUML을 HTML로 변환하는 기능 (선택사항)"""
    try:
        import subprocess
        print("\nHTML 변환을 시도합니다...")
        result = subprocess.run(['plantuml', '-tsvg', 'merged_sequence_diagrams.txt'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("SVG 파일이 생성되었습니다.")
        else:
            print("PlantUML이 설치되지 않았거나 오류가 발생했습니다.")
            print("PlantUML을 설치하려면: sudo apt-get install plantuml")
    except FileNotFoundError:
        print("PlantUML이 설치되지 않았습니다.")
        print("PlantUML을 설치하려면: sudo apt-get install plantuml")

if __name__ == "__main__":
    merge_sequence_diagrams()
    
    # 사용자에게 HTML 변환 여부 묻기
    response = input("\nHTML로 변환하시겠습니까? (y/n): ").lower().strip()
    if response in ['y', 'yes', '예']:
        create_html_output()
    
    print("\n스크립트 실행이 완료되었습니다.")
