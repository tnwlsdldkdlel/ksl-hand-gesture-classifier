#!/usr/bin/env python3
"""
KSL 손동작 분류 모델 재생성 스크립트
구조+가중치가 포함된 전체 모델로 저장하여 TF.js 변환 시 오류를 방지합니다.
"""

import argparse
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path):
    """데이터 로드"""
    print(f"데이터 로드 중: {data_path}")
    data = np.load(data_path)
    X = data['X'].astype('float32')
    y = data['y'].astype('int32')
    print(f"데이터 형태: X={X.shape}, y={y.shape}")
    return X, y

def create_model():
    """모델 생성"""
    print("\n모델 생성 중...")
    model = keras.Sequential([
        layers.Input(shape=(63,), name='input'),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dense(5, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n모델 구조:")
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    """모델 학습"""
    print(f"\n모델 학습 시작...")
    print(f"학습 데이터: {X_train.shape[0]}개")
    print(f"검증 데이터: {X_val.shape[0]}개")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # EarlyStopping 콜백
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def verify_no_lambda_layers(model):
    """모델에 Lambda 레이어가 없는지 확인"""
    lambda_layers = [layer for layer in model.layers if isinstance(layer, layers.Lambda)]
    if lambda_layers:
        raise ValueError(
            f"모델에 Lambda 레이어가 발견되었습니다: {[l.name for l in lambda_layers]}\n"
            "TF.js는 Lambda 레이어를 지원하지 않습니다. 모델 구조를 수정해주세요."
        )
    print("✅ Lambda 레이어 없음 확인 (TF.js 호환)")

def save_full_model(model, output_path):
    """전체 모델 저장 (구조+가중치)"""
    print(f"\n전체 모델 저장 중: {output_path}")
    
    # Lambda 레이어 확인
    verify_no_lambda_layers(model)
    
    # model.save()는 구조와 가중치를 모두 저장합니다
    model.save(output_path)
    print(f"저장 완료: {output_path}")
    
    # 파일 크기 확인
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"파일 크기: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description='KSL 손동작 분류 모델 재생성')
    parser.add_argument('--data', type=str, default='data/ksl_landmarks.npz',
                        help='학습 데이터 파일 (default: data/ksl_landmarks.npz)')
    parser.add_argument('--output', type=str, default='model_full.h5',
                        help='출력 모델 파일 (default: model_full.h5)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='학습 에포크 수 (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='배치 크기 (default: 64)')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='테스트 세트 비율 (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='검증 세트 비율 (default: 0.15)')
    
    args = parser.parse_args()
    
    # 데이터 로드
    X, y = load_data(args.data)
    
    # 데이터 정규화 (0~1)
    print("\n데이터 정규화 중...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # 데이터 분할: 학습/검증/테스트
    print("\n데이터 분할 중...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"학습 세트: {X_train.shape[0]}개")
    print(f"검증 세트: {X_val.shape[0]}개")
    print(f"테스트 세트: {X_test.shape[0]}개")
    
    # 모델 생성
    model = create_model()
    
    # 모델 학습
    history = train_model(model, X_train, y_train, X_val, y_val, 
                         epochs=args.epochs, batch_size=args.batch_size)
    
    # 테스트 세트 평가
    print("\n테스트 세트 평가 중...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 전체 모델 저장 (구조+가중치)
    save_full_model(model, args.output)
    
    print("\n✅ 모델 재생성 완료!")
    print(f"다음 단계: TF.js 변환")
    print(f"  python3 scripts/convert_to_tfjs.py --model {args.output} --output public/model")

if __name__ == '__main__':
    main()

