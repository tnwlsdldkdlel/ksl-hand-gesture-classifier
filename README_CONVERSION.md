# 모델 변환 가이드

## 현재 상황

ONNX 변환 시 protobuf 버전 충돌 문제가 발생했습니다:
- TensorFlow 2.20.0은 protobuf>=5.28.0 요구
- tf2onnx는 protobuf~=3.20 요구
- 두 요구사항이 호환되지 않음

## 해결 방법

### 옵션 1: TensorFlow.js 변환 (권장)

```bash
# tensorflowjs 설치 (시간이 걸릴 수 있음)
pip3 install tensorflowjs

# 변환 실행
python3 scripts/convert_to_tfjs.py --model model.h5 --output public/model
```

### 옵션 2: 모델을 그대로 사용

Keras 모델(.h5)을 Python 백엔드에서 직접 사용:
- Flask/FastAPI 서버에서 모델 로드
- REST API로 추론 제공
- 웹에서 API 호출

### 옵션 3: 다른 환경에서 ONNX 변환

별도의 가상환경에서:
```bash
python3 -m venv venv_onnx
source venv_onnx/bin/activate
pip install tensorflow==2.13.0 onnx keras2onnx
python scripts/convert_to_onnx_simple.py --model model.h5 --output public/model/model.onnx
```

## 현재 모델 정보

- 모델 파일: `model.h5`
- 입력: (63,) - 21포인트 × 3차원
- 출력: (5,) - KSL_1~KSL_5 클래스
- 정확도: 100% (테스트 세트)

