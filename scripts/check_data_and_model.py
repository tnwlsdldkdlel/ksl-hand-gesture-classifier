#!/usr/bin/env python3
"""
데이터 클래스 분포 및 모델 출력 레이어 확인 스크립트
"""

import numpy as np
import os
import sys
from collections import Counter

def check_data_classes(data_path):
    """데이터 파일의 클래스 분포 확인"""
    print(f"\n{'='*60}")
    print(f"데이터 파일 확인: {data_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return None
    
    try:
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        print(f"\n데이터 형태:")
        print(f"  X (입력): {X.shape}")
        print(f"  y (레이블): {y.shape}")
        
        # 클래스 분포 확인
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        print(f"\n클래스 분포:")
        print(f"  총 샘플 수: {total_samples}")
        print(f"  고유 클래스 수: {len(unique_classes)}")
        print(f"\n  클래스별 샘플 수:")
        
        class_mapping = {0: 'KSL_1', 1: 'KSL_2', 2: 'KSL_3', 3: 'KSL_4', 4: 'KSL_5'}
        
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total_samples) * 100
            class_name = class_mapping.get(cls, f'Unknown_{cls}')
            print(f"    클래스 {cls} ({class_name}): {count:,}개 ({percentage:.2f}%)")
        
        # 누락된 클래스 확인
        expected_classes = set(range(5))  # 0~4
        actual_classes = set(unique_classes)
        missing_classes = expected_classes - actual_classes
        
        if missing_classes:
            print(f"\n  ⚠️  누락된 클래스:")
            for cls in sorted(missing_classes):
                print(f"    클래스 {cls} ({class_mapping.get(cls, f'Unknown_{cls}')}): 데이터 없음")
        else:
            print(f"\n  ✅ 모든 클래스(0~4)가 데이터에 포함되어 있습니다.")
        
        # 클래스 불균형 확인
        if len(counts) > 1:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"\n  클래스 불균형 비율: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 2.0:
                print(f"  ⚠️  클래스 불균형이 심각합니다. 데이터 증강 또는 샘플링이 필요합니다.")
            elif imbalance_ratio > 1.5:
                print(f"  ⚠️  클래스 불균형이 있습니다. 데이터 증강을 고려하세요.")
            else:
                print(f"  ✅ 클래스 분포가 비교적 균형적입니다.")
        
        return {
            'X': X,
            'y': y,
            'unique_classes': unique_classes,
            'counts': counts,
            'missing_classes': missing_classes
        }
        
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_model_output(model_path):
    """모델 파일의 출력 레이어 확인"""
    print(f"\n{'='*60}")
    print(f"모델 파일 확인: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    try:
        from tensorflow import keras
        
        print(f"\n모델 로드 중...")
        model = keras.models.load_model(model_path)
        
        print(f"\n모델 구조:")
        model.summary()
        
        # 출력 레이어 확인
        output_layer = model.layers[-1]
        
        # 출력 형태 확인 (여러 방법 시도)
        try:
            output_shape = output_layer.output_shape
        except AttributeError:
            try:
                output_shape = output_layer.output_shape_
            except AttributeError:
                try:
                    output_shape = model.output_shape
                except:
                    output_shape = "확인 불가"
        
        print(f"\n출력 레이어 정보:")
        print(f"  레이어 이름: {output_layer.name}")
        print(f"  레이어 타입: {type(output_layer).__name__}")
        print(f"  출력 형태: {output_shape}")
        
        # 출력 클래스 수 확인
        if hasattr(output_layer, 'units'):
            num_classes = output_layer.units
            print(f"  출력 클래스 수: {num_classes}")
            
            if num_classes == 5:
                print(f"  ✅ 모델이 5개 클래스를 출력하도록 설정되어 있습니다.")
            else:
                print(f"  ⚠️  모델이 {num_classes}개 클래스를 출력합니다. (예상: 5개)")
        
        # 활성화 함수 확인
        if hasattr(output_layer, 'activation'):
            activation = output_layer.activation
            if hasattr(activation, '__name__'):
                activation_name = activation.__name__
            else:
                activation_name = str(activation)
            print(f"  활성화 함수: {activation_name}")
            
            if activation_name == 'softmax':
                print(f"  ✅ Softmax 활성화 함수 사용 (다중 클래스 분류에 적합)")
            else:
                print(f"  ⚠️  Softmax가 아닌 활성화 함수 사용: {activation_name}")
        
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 클래스 분포 및 모델 출력 확인')
    parser.add_argument('--data', type=str, default='data/ksl_landmarks.npz',
                        help='데이터 파일 경로 (default: data/ksl_landmarks.npz)')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='모델 파일 경로 (default: model.h5)')
    parser.add_argument('--check-all-models', action='store_true',
                        help='모든 모델 파일 확인 (model*.h5)')
    
    args = parser.parse_args()
    
    # 데이터 확인
    data_info = check_data_classes(args.data)
    
    # 모델 확인
    if args.check_all_models:
        # 모든 모델 파일 확인
        model_files = [f for f in os.listdir('.') if f.endswith('.h5') and f.startswith('model')]
        if model_files:
            print(f"\n{'='*60}")
            print(f"모든 모델 파일 확인")
            print(f"{'='*60}")
            for model_file in sorted(model_files):
                check_model_output(model_file)
        else:
            print(f"\n⚠️  모델 파일을 찾을 수 없습니다.")
    else:
        check_model_output(args.model)
    
    # 종합 분석
    print(f"\n{'='*60}")
    print(f"종합 분석")
    print(f"{'='*60}")
    
    if data_info:
        missing_classes = data_info['missing_classes']
        if missing_classes:
            print(f"\n❌ 문제 발견:")
            print(f"  데이터에 {len(missing_classes)}개 클래스가 누락되어 있습니다.")
            print(f"  누락된 클래스: {sorted(missing_classes)}")
            print(f"\n  해결 방법:")
            print(f"    1. 누락된 클래스의 데이터 수집")
            print(f"    2. 데이터 증강 (조명, 각도, 배경 변화)")
            print(f"    3. 클래스 불균형 해결 (각 클래스 비율 맞추기)")
        else:
            print(f"\n✅ 데이터에 모든 클래스(0~4)가 포함되어 있습니다.")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()

