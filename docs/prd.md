# 📄 KSL 손동작 분류 모델 PRD (Python 학습/변환)

## 1) 개요
- 목표: KSL 숫자 1~5 손동작을 MediaPipe 랜드마크(21포인트) 기반 5클래스 분류.
- 산출물: `model.h5`(Keras), TF.js 변환물(`public/model/model.json`, `group1-shard*.bin`).

## 2) 입력/라벨 정의
- 입력: 21포인트 × (x,y,z) = 63차원 `float32` 벡터.
- 라벨: KSL_1~KSL_5 → 정수 0~4 매핑.

## 3) 데이터 요구사항
- 전처리: 0~1 정규화(필요 시 손 중심/스케일 정규화 옵션).
- 분할: 학습/검증/테스트 ≈ 70/15/15 (stratify).
- 포맷: `data/ksl_landmarks.npz`  
  - `X`: (N,63) `float32`  
  - `y`: (N,) `int` 0~4

## 4) 모델/학습 스펙
- 모델(예시 MLP): Input 63 → Dense 64 ReLU → Dense 32 ReLU → Dense 5 Softmax.
- 손실/옵티마이저: `sparse_categorical_crossentropy` / `adam`.
- 콜백: EarlyStopping(`val_loss`, `patience=5`, `restore_best_weights=True`).
- 하이퍼파라미터: epochs 30~50, batch_size 32~128 범위 탐색.

## 5) 검증 기준
- 지표: 정확도, confusion matrix, 클래스별 precision/recall/F1.
- 목표: 검증/테스트 정확도 90% 이상(데이터 품질 따라 조정).
- 과적합 대응: EarlyStopping, 필요 시 드롭아웃/가중치감쇠, 좌표 노이즈 증강.

## 6) 산출물
- 학습 모델: `model.h5`
- TF.js 변환물: `public/model/model.json` + `group1-shard*.bin`
- 변환 명령:
  ```bash
  npm install -g @tensorflow/tfjs-cli   # 1회
  tensorflowjs_converter --input_format=keras model.h5 ./public/model
  ```

## 7) 실행/재현
- 학습 스크립트 예: `python scripts/train_ksl.py --data data/ksl_landmarks.npz --epochs 50 --batch-size 64 --model-out model.h5`
- 종속성: Python 3.x, `tensorflow`, `numpy`, `scikit-learn`.

## 8) 리스크 및 대응
- 데이터 부족/불균형: 증강(좌표 노이즈, 약간의 스케일), stratify 분할.
- 클래스/입력 불일치: 라벨 순서(0~4)와 앱 매핑(KSL_1~5) 일관성 검증.
- 성능 미달: 은닉층/노드 수 조정, 학습률/스케줄 튜닝.

## 9) 연동 확인
- 앱은 `/model/model.json`을 로드하므로 변환 결과를 `public/model/`에 배치.
- 학습 시 전처리/라벨 순서를 추론 단계에서도 동일하게 적용.