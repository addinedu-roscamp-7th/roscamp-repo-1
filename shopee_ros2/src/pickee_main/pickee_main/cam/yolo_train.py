from ultralytics import YOLO

if __name__ == '__main__':
    # 데이터 셋 경로 및 설정
    model = YOLO('yolov8n.pt')
    
    # 오탐지 방지를 위한 개선된 학습 설정
    model.train(
        data='employee_data/data.yaml',
        epochs=100,  # 에포크 증가 (50 -> 100)
        imgsz=640,
        batch=16,
        patience=15,  # Early stopping patience
        
        # 데이터 증강 (Augmentation) 강화 - 다양한 환경에서의 일반화
        hsv_h=0.015,  # 색조 변화
        hsv_s=0.7,    # 채도 변화
        hsv_v=0.4,    # 명도 변화
        degrees=10,   # 회전
        translate=0.1,  # 이동
        scale=0.5,    # 스케일
        flipud=0.0,   # 상하 반전 비활성화 (사람 얼굴은 항상 위쪽)
        fliplr=0.5,   # 좌우 반전
        mosaic=1.0,   # 모자이크 증강
        mixup=0.1,    # MixUp 증강
        
        # 학습 파라미터
        lr0=0.001,    # 초기 학습률 낮춤 (과적합 방지)
        lrf=0.01,     # 최종 학습률
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 손실 함수 가중치 조정
        box=7.5,      # 박스 손실
        cls=0.5,      # 클래스 손실
        dfl=1.5,      # DFL 손실
        
        # 기타 설정
        name='yolo_employee_model_v2',
        pretrained=True,
        optimizer='AdamW',  # AdamW 옵티마이저 사용
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # 단일 클래스(본인만) 학습
        rect=False,
        cos_lr=True,  # Cosine 학습률 스케줄러
        close_mosaic=10,  # 마지막 10 에포크는 모자이크 비활성화
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        save=True,
        save_period=-1,
        cache=False,
        device='0',  # GPU 사용 (없으면 'cpu'로 변경)
        workers=8,
        project=None,
        exist_ok=False,
        resume=False,
        nbs=64,
        plots=True
    )