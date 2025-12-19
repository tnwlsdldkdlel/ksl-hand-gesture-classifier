#!/usr/bin/env python3
"""
KSL 손동작 분류 모델 학습 스크립트 (개선 버전)
5개 클래스(KSL_1~KSL_5) 균형 학습 및 검증
각 클래스가 모두 학습되도록 강화
"""

import argparse
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path):
    """데이터 로드"""
    print(f"\n{'='*60}")
    print(f"데이터 로드 중: {data_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    data = np.load(data_path)
    X = data['X'].astype('float32')
    y = data['y'].astype('int32')
    
    print(f"\n데이터 형태:")
    print(f"  X (입력): {X.shape}")
    print(f"  y (레이블): {y.shape}")
    
    # 클래스 분포 확인
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    print(f"\n클래스 분포:")
    class_mapping = {0: 'KSL_1', 1: 'KSL_2', 2: 'KSL_3', 3: 'KSL_4', 4: 'KSL_5'}
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_samples) * 100
        class_name = class_mapping.get(cls, f'Unknown_{cls}')
        print(f"    클래스 {cls} ({class_name}): {count:,}개 ({percentage:.2f}%)")
    
    # 누락된 클래스 확인
    expected_classes = set(range(5))
    actual_classes = set(unique_classes)
    missing_classes = expected_classes - actual_classes
    
    if missing_classes:
        raise ValueError(f"❌ 누락된 클래스: {sorted(missing_classes)}. 모든 클래스(0~4)가 필요합니다!")
    
    # 클래스 불균형 확인
    if len(counts) > 1:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        if imbalance_ratio > 1.5:
            print(f"\n  ⚠️  클래스 불균형 비율: {imbalance_ratio:.2f}:1")
            print(f"  경고: 클래스 불균형이 있습니다. 학습에 영향을 줄 수 있습니다.")
    
    return X, y

def create_model(learning_rate=0.001):
    """모델 생성"""
    print(f"\n{'='*60}")
    print(f"모델 생성 중...")
    print(f"{'='*60}")
    print(f"Learning rate: {learning_rate}")
    
    # 가중치 초기화를 명시적으로 설정
    model = keras.Sequential([
        layers.Input(shape=(63,), name='input'),
        layers.Dense(64, activation='relu', name='dense_1',
                    kernel_initializer=glorot_uniform(seed=42)),
        layers.Dense(32, activation='relu', name='dense_2',
                    kernel_initializer=glorot_uniform(seed=42)),
        layers.Dense(5, activation='softmax', name='output',
                    kernel_initializer=glorot_uniform(seed=42))
    ])
    
    # Adam 옵티마이저에 learning rate 명시
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n모델 구조:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
               class_weight=None):
    """모델 학습"""
    print(f"\n{'='*60}")
    print(f"모델 학습 시작...")
    print(f"{'='*60}")
    print(f"학습 데이터: {X_train.shape[0]}개")
    print(f"검증 데이터: {X_val.shape[0]}개")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # EarlyStopping 콜백 (더 긴 patience)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001
    )
    
    # ModelCheckpoint 콜백
    checkpoint = keras.callbacks.ModelCheckpoint(
        'model_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        class_weight=class_weight,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, output_dir='.'):
    """모델 평가 및 confusion matrix 생성"""
    print(f"\n{'='*60}")
    print(f"모델 평가 중...")
    print(f"{'='*60}")
    
    # 테스트 세트 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 예측
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification Report
    class_names = ['KSL_1', 'KSL_2', 'KSL_3', 'KSL_4', 'KSL_5']
    print(f"\n{'='*60}")
    print(f"Classification Report")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    print(f"\n{'='*60}")
    print(f"Confusion Matrix")
    print(f"{'='*60}")
    print(cm)
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion Matrix 저장: {cm_path}")
    plt.close()
    
    # 클래스별 정확도 상세 분석
    print(f"\n{'='*60}")
    print(f"클래스별 정확도 상세 분석")
    print(f"{'='*60}")
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum((y_pred_classes[class_mask] == i))
            class_total = np.sum(class_mask)
            class_accuracy = class_correct / class_total
            class_accuracies[class_name] = class_accuracy
            
            # 예측 확률 분포 확인
            class_probs = y_pred[class_mask, i]
            avg_prob = np.mean(class_probs)
            max_prob = np.max(class_probs)
            min_prob = np.min(class_probs)
            
            print(f"\n  {class_name} (클래스 {i}):")
            print(f"    정확도: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
            print(f"    올바른 예측: {class_correct}/{class_total}")
            print(f"    평균 예측 확률: {avg_prob:.4f}")
            print(f"    최대 예측 확률: {max_prob:.4f}")
            print(f"    최소 예측 확률: {min_prob:.4f}")
            
            # 잘못 예측된 경우 분석
            wrong_mask = class_mask & (y_pred_classes != i)
            if np.sum(wrong_mask) > 0:
                wrong_preds = y_pred_classes[wrong_mask]
                wrong_counts = np.bincount(wrong_preds, minlength=5)
                print(f"    잘못 예측된 경우: {np.sum(wrong_mask)}개")
                for j, count in enumerate(wrong_counts):
                    if count > 0:
                        print(f"      → {class_names[j]}로 {count}개 잘못 예측")
        else:
            print(f"\n  {class_name} (클래스 {i}): 데이터 없음")
            class_accuracies[class_name] = 0.0
    
    # 모든 클래스가 90% 이상인지 확인
    print(f"\n{'='*60}")
    print(f"클래스별 정확도 요약")
    print(f"{'='*60}")
    all_above_90 = True
    for class_name, acc in class_accuracies.items():
        status = "✅" if acc >= 0.90 else "❌"
        print(f"  {status} {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        if acc < 0.90:
            all_above_90 = False
    
    if all_above_90:
        print(f"\n✅ 모든 클래스가 90% 이상의 정확도를 달성했습니다!")
    else:
        print(f"\n⚠️  일부 클래스가 90% 미만입니다. 추가 학습이 필요할 수 있습니다.")
    
    return test_accuracy, cm, class_accuracies

def save_model(model, output_path):
    """모델 저장"""
    print(f"\n{'='*60}")
    print(f"모델 저장 중: {output_path}")
    print(f"{'='*60}")
    
    model.save(output_path)
    print(f"✅ 모델 저장 완료: {output_path}")
    
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"파일 크기: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description='KSL 손동작 분류 모델 학습 (개선 버전)')
    parser.add_argument('--data', type=str, default='data/ksl_landmarks.npz',
                        help='학습 데이터 파일 (default: data/ksl_landmarks.npz)')
    parser.add_argument('--output', type=str, default='model.h5',
                        help='출력 모델 파일 (default: model.h5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수 (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='배치 크기 (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='테스트 세트 비율 (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='검증 세트 비율 (default: 0.2)')
    parser.add_argument('--class-weight-boost', type=float, default=1.0,
                        help='클래스 1/2/3/4 가중치 부스트 (default: 1.0, 예: 1.2)')
    
    args = parser.parse_args()
    
    # 데이터 로드
    X, y = load_data(args.data)
    
    # 추가 정규화 없이 MediaPipe 출력 그대로 사용 (프론트와 일치)
    print(f"\n{'='*60}")
    print(f"추가 스케일링 없이 원본 입력 사용 (프론트 전처리와 일치)")
    print(f"{'='*60}")
    print(f"입력 값 범위 확인: X 범위 [{X.min():.4f}, {X.max():.4f}]")
    
    # 데이터 분할: 학습/검증/테스트
    print(f"\n{'='*60}")
    print(f"데이터 분할 중...")
    print(f"{'='*60}")
    # 먼저 테스트 세트 분리
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    # 그 다음 학습/검증 세트 분리
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"학습 세트: {X_train.shape[0]}개")
    print(f"검증 세트: {X_val.shape[0]}개")
    print(f"테스트 세트: {X_test.shape[0]}개")
    
    # 각 세트의 클래스 분포 확인
    print(f"\n학습 세트 클래스 분포:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(train_unique, train_counts):
        print(f"  클래스 {cls}: {count}개")
    
    # 모델 생성
    model = create_model(learning_rate=args.learning_rate)
    
    # 클래스 가중치 설정 (필요 시)
    class_weight = None
    if args.class_weight_boost > 1.0:
        b = args.class_weight_boost
        class_weight = {0: 1.0, 1: b, 2: b, 3: b, 4: 1.0}
        print(f"\n클래스 가중치 적용: {class_weight}")

    # 모델 학습
    history = train_model(model, X_train, y_train, X_val, y_val, 
                         epochs=args.epochs, batch_size=args.batch_size,
                         class_weight=class_weight)
    
    # 모델 평가
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    test_accuracy, cm, class_accuracies = evaluate_model(model, X_test, y_test, output_dir)
    
    # 모델 저장
    save_model(model, args.output)
    
    print(f"\n{'='*60}")
    print(f"✅ 학습 완료!")
    print(f"{'='*60}")
    print(f"모델 파일: {args.output}")
    print(f"테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\n다음 단계: TensorFlow.js 변환")
    print(f"  python3 scripts/convert_to_tfjs.py --model {args.output} --output public/model")

if __name__ == '__main__':
    main()
