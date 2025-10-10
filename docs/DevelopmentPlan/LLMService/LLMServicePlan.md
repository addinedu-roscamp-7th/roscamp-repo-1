# Shopee LLM Service 개발 계획

## 1단계: 프로젝트 기반 설정 (Foundation)
- **목표**: LLM 서비스 제공을 위한 기본 환경 구축
- **세부 작업**:
  1. **프로젝트 초기화**: Python, FastAPI 기반 프로젝트 구조 생성
  2. **LLM 라이브러리 설치**: Hugging Face `transformers`, `PyTorch`/`TensorFlow` 등 라이브러리 설치
  3. **모델 로드**: 사용할 사전 훈련된 언어 모델(e.g., Ko-BERT, GPT)을 로드하고 기본 추론 테스트

## 2단계: API 구현 (API Implementation)
- **목표**: `Main_vs_LLM.md` 명세에 따른 API 엔드포인트 구현
- **세부 작업**:
  1. **`/search_query` 엔드포인트**: 자연어 텍스트를 받아 SQL 쿼리를 반환하는 API 구현
  2. **`/intent_detection` 엔드포인트**: 자연어 텍스트를 받아 의도(intent)와 개체(entities)를 반환하는 API 구현

## 3단계: 핵심 로직 구현 및 고도화 (Core Logic & Enhancement)
- **목표**: 실제 비즈니스 요구사항에 맞는 자연어 처리 로직 구현
- **세부 작업**:
  1. **SQL 생성 로직**: 상품 테이블(product) 구조에 맞춰, 자연어에서 키워드를 추출하고 SQL `WHERE` 절을 생성하는 로직 구현
  2. **의도 분석 로직**: '가져다줘', '찾아줘' 등 주요 발화 의도를 분류하고, 상품명, 수량 등 주요 개체를 추출하는 로직 구현
  3. **(선택) 모델 미세조정(Fine-tuning)**: Shopee 도메인에 특화된 데이터셋을 구축하여, 모델의 성능을 향상시키는 미세조정 작업 진행
