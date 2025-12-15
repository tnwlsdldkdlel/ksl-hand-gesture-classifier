# KSL 손동작 분류 모델

MediaPipe 기반 KSL 숫자 손동작 인식 모델

## 개요

KSL(한국수화언어) 숫자 1~5 손동작을 MediaPipe 랜드마크(21포인트) 기반으로 분류하는 딥러닝 모델입니다.

## 주요 기능

- MediaPipe 손 랜드마크(21포인트 × 3차원 = 63차원) 입력
- KSL 숫자 1~5 손동작 분류 (5클래스)
- TensorFlow/Keras 기반 MLP 모델
- TensorFlow.js 변환 지원

## 프로젝트 구조

```
ksl-hand-gesture-classifier/
├── data/                    # 데이터 파일
│   └── ksl_landmarks.npz   # 학습 데이터
├── docs/                    # 문서
│   └── prd.md              # 프로젝트 요구사항 문서
├── scripts/                 # 스크립트
│   ├── train_ksl.py        # 모델 학습 스크립트
│   ├── generate_sample_data.py  # 샘플 데이터 생성
│   └── convert_to_tfjs.py  # TF.js 변환 스크립트
├── public/                  # 웹 배포용
│   └── model/              # TF.js 변환 모델
├── model.h5                 # 학습된 Keras 모델
└── requirements.txt         # Python 종속성
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 준비

```bash
# 샘플 데이터 생성 (테스트용)
python scripts/generate_sample_data.py --samples-per-class 200

# 또는 실제 데이터를 data/ksl_landmarks.npz 형식으로 준비
```

### 2. 모델 학습

```bash
python scripts/train_ksl.py \
  --data data/ksl_landmarks.npz \
  --epochs 50 \
  --batch-size 64 \
  --model-out model.h5
```

### 3. TensorFlow.js 변환

```bash
python scripts/convert_to_tfjs.py \
  --model model.h5 \
  --output public/model
```

## 모델 정보

- **입력**: 63차원 벡터 (21포인트 × x,y,z)
- **출력**: 5클래스 (KSL_1 ~ KSL_5)
- **아키텍처**: MLP (63 → 64 → 32 → 5)
- **정확도**: 테스트 세트 기준 90% 이상 목표

## 요구사항

- Python 3.x
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0

## 라이선스

[라이선스 정보 추가]

## 기여

기여 가이드는 [`.cursor/CONTRIBUTING.md`](.cursor/CONTRIBUTING.md)를 참조하세요.

