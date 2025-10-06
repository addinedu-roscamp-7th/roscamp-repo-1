#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시퀀스 다이어그램 XML 파일들을 SC 번호와 함께 하나의 텍스트 파일로 병합하는 스크립트
"""

import os
import re
import glob
from pathlib import Path

def get_file_order():
    """파일들을 올바른 순서로 정렬합니다."""
    # 파일명에서 순서를 추출하여 정렬
    pattern = r'SC_(\d+)_(\d+)\.drawio\.xml'
    files = []
    
    for file_path in glob.glob("SC_*.drawio.xml"):
        match = re.search(pattern, file_path)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            files.append((major, minor, file_path))
    
    # SC_04는 별도로 처리 (숫자가 없음)
    for file_path in glob.glob("SC_04.drawio.xml"):
        files.append((4, 0, file_path))
    
    # 정렬: (major, minor) 순서로
    files.sort(key=lambda x: (x[0], x[1]))
    return [file_path for _, _, file_path in files]

def extract_sc_number(filename):
    """파일명에서 SC 번호를 추출합니다."""
    # SC_01_1.drawio.xml -> SC_01_1
    # SC_04.drawio.xml -> SC_04
    match = re.match(r'(SC_\d+(?:_\d+)?)\.drawio\.xml', filename)
    if match:
        return match.group(1)
    return filename.replace('.drawio.xml', '')

def merge_xml_files():
    """XML 파일들을 SC 번호와 함께 병합합니다."""
    print("XML 파일 병합을 시작합니다...")
    
    # 현재 디렉토리에서 SC_로 시작하는 XML 파일들 찾기
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 파일 순서 정렬
    ordered_files = get_file_order()
    
    if not ordered_files:
        print("SC_로 시작하는 XML 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 파일들: {ordered_files}")
    
    # 병합된 내용 저장
    merged_content = []
    merged_content.append("=" * 80)
    merged_content.append("시퀀스 다이어그램 XML 파일 병합본")
    merged_content.append("=" * 80)
    merged_content.append("")
    
    # 각 파일 처리
    for i, file_path in enumerate(ordered_files):
        print(f"처리 중: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # SC 번호 추출
            sc_number = extract_sc_number(file_path)
            
            # 섹션 헤더 추가
            merged_content.append("=" * 80)
            merged_content.append(f"SC 번호: {sc_number}")
            merged_content.append(f"파일명: {file_path}")
            merged_content.append("=" * 80)
            merged_content.append("")
            
            # XML 내용 추가
            merged_content.append(content)
            
            # 파일 간 구분을 위한 빈 라인 추가
            if i < len(ordered_files) - 1:
                merged_content.append("")
                merged_content.append("")
                merged_content.append("-" * 80)
                merged_content.append("")
            
        except Exception as e:
            print(f"파일 {file_path} 처리 중 오류: {e}")
            continue
    
    # 병합된 내용을 파일로 저장
    output_file = "merged_xml_with_sc_numbers.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merged_content))
    
    print(f"\n병합 완료! 결과 파일: {output_file}")
    print(f"총 {len(ordered_files)}개 파일이 병합되었습니다.")
    
    # 파일별 요약 정보 출력
    print("\n파일별 요약:")
    for i, file_path in enumerate(ordered_files, 1):
        sc_number = extract_sc_number(file_path)
        print(f"{i:2d}. {sc_number} - {file_path}")

def create_summary():
    """병합된 파일의 요약 정보를 생성합니다."""
    summary_file = "xml_merge_summary.txt"
    
    ordered_files = get_file_order()
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("시퀀스 다이어그램 XML 파일 병합 요약\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"총 파일 수: {len(ordered_files)}개\n\n")
        
        f.write("파일 목록:\n")
        f.write("-" * 30 + "\n")
        
        for i, file_path in enumerate(ordered_files, 1):
            sc_number = extract_sc_number(file_path)
            f.write(f"{i:2d}. {sc_number} - {file_path}\n")
        
        f.write("\n병합된 파일: merged_xml_with_sc_numbers.txt\n")
    
    print(f"요약 파일 생성: {summary_file}")

if __name__ == "__main__":
    merge_xml_files()
    create_summary()
    print("\n스크립트 실행이 완료되었습니다.")
