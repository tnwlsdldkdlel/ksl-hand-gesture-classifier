#!/usr/bin/env python3
"""
Lambda 레이어를 제거하고 TF.js 호환 모델로 재생성하는 스크립트
"""

import argparse
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def remove_lambda_layers(model_path, output_path):
    """Lambda 레이어를 제거한 새 모델 생성"""
    print(f"모델 로드 중: {model_path}")
    original_model = keras.models.load_model(model_path)
    
    print("원본 모델 구조:")
    original_model.summary()
    
    # Lambda 레이어를 제거한 새 모델 생성
    print("\nLambda 레이어 제거된 새 모델 생성 중...")
    
    # 원본 모델의 구조를 분석하여 새 모델 생성
    input_shape = None
    dense_layers = []
    
    for layer in original_model.layers:
        if isinstance(layer, layers.InputLayer):
            input_shape = layer.input_shape[1:]  # (None, 63) -> (63,)
        elif isinstance(layer, layers.Lambda):
            print(f"  Lambda 레이어 '{layer.name}' 제거됨")
            continue
        elif isinstance(layer, layers.Dense):
            dense_layers.append(layer)
    
    # 새 모델 생성
    new_model = keras.Sequential()
    if input_shape:
        new_model.add(layers.Input(shape=input_shape))
    
    for layer in dense_layers:
        # 새로운 Dense 레이어 생성 (설정 복사)
        new_layer = layers.Dense(
            units=layer.units,
            activation=layer.activation,
            name=layer.name
        )
        new_model.add(new_layer)
    
    # 모델 빌드
    new_model.build(input_shape=(None, input_shape[0] if input_shape else 63))
    
    # 가중치 복사
    print("\n가중치 복사 중...")
    for new_layer in new_model.layers:
        if isinstance(new_layer, layers.Dense):
            # 원본 모델에서 동일한 이름의 레이어 찾기
            for orig_layer in original_model.layers:
                if isinstance(orig_layer, layers.Dense) and orig_layer.name == new_layer.name:
                    new_layer.set_weights(orig_layer.get_weights())
                    print(f"  {new_layer.name} 가중치 복사 완료")
                    break
    
    # 모델 컴파일
    new_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n새 모델 구조:")
    new_model.summary()
    
    # 모델 저장
    print(f"\n모델 저장 중: {output_path}")
    new_model.save(output_path)
    print("저장 완료!")
    
    return new_model


def main():
    parser = argparse.ArgumentParser(description='Lambda 레이어 제거 및 TF.js 호환 모델 생성')
    parser.add_argument('--input', type=str, default='model.h5',
                        help='입력 모델 파일 (default: model.h5)')
    parser.add_argument('--output', type=str, default='model_fixed.h5',
                        help='출력 모델 파일 (default: model_fixed.h5)')
    
    args = parser.parse_args()
    
    remove_lambda_layers(args.input, args.output)
    print(f"\n✅ 완료! TF.js 호환 모델: {args.output}")


if __name__ == '__main__':
    main()

