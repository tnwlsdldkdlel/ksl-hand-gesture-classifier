#!/usr/bin/env python3
"""
Keras 모델을 TF.js 형식으로 변환하는 스크립트
"""

import argparse
import json
import os
import sys
import subprocess

def check_and_install_tensorflowjs():
    """tensorflowjs 설치 확인 및 설치"""
    try:
        import tensorflowjs as tfjs
        return tfjs
    except ImportError:
        print("tensorflowjs가 설치되지 않았습니다.")
        print("설치 중... (시간이 걸릴 수 있습니다)")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "tensorflowjs"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            import tensorflowjs as tfjs
            print("설치 완료!")
            return tfjs
        except subprocess.CalledProcessError as e:
            print(f"설치 실패: {e}")
            print("\n수동 설치 명령어:")
            print("  pip3 install tensorflowjs")
            sys.exit(1)


def verify_weights_match_layers(model_json_path):
    """가중치 매니페스트의 레이어 이름이 모델 구조와 일치하는지 확인"""
    try:
        with open(model_json_path, 'r') as f:
            data = json.load(f)
        
        # 레이어 이름 추출 (InputLayer 제외)
        layers = data['modelTopology']['model_config']['config']['layers']
        layer_names = {layer['config'].get('name') for layer in layers 
                       if layer.get('class_name') != 'InputLayer'}
        
        # 가중치 매니페스트에서 레이어 이름 추출
        manifest = data['weightsManifest'][0]
        weight_layer_names = set()
        for weight in manifest['weights']:
            # sequential/dense_1/kernel -> dense_1
            parts = weight['name'].split('/')
            if len(parts) >= 2:
                weight_layer_names.add(parts[1])
        
        # 일치 확인
        if layer_names != weight_layer_names:
            print(f"\n  ⚠️  경고: 레이어 이름 불일치 발견!")
            print(f"     모델 구조 레이어: {sorted(layer_names)}")
            print(f"     가중치 매니페스트 레이어: {sorted(weight_layer_names)}")
            print(f"     누락된 레이어: {sorted(layer_names - weight_layer_names)}")
            print(f"     추가된 레이어: {sorted(weight_layer_names - layer_names)}")
            return False
        else:
            print(f"  ✅ 가중치-레이어 이름 일치 확인: {sorted(layer_names)}")
            return True
    except Exception as e:
        print(f"  ⚠️  가중치 검증 중 오류: {e}")
        return False


def fix_inputlayer_in_model_json(model_json_path):
    """TF.js 호환성을 위해 InputLayer의 batch_shape를 batchInputShape로 변환"""
    try:
        with open(model_json_path, 'r') as f:
            data = json.load(f)
        
        layers = data['modelTopology']['model_config']['config']['layers']
        modified = False
        
        for layer in layers:
            if layer.get('class_name') == 'InputLayer':
                config = layer.get('config', {})
                if 'batch_shape' in config and 'batchInputShape' not in config:
                    # batch_shape를 batchInputShape로 변환
                    batch_shape = config.pop('batch_shape')
                    config['batchInputShape'] = batch_shape
                    
                    # inputShape가 있으면 제거 (TF.js는 batchInputShape 또는 inputShape 중 하나만 허용)
                    if 'inputShape' in config:
                        config.pop('inputShape')
                    
                    modified = True
                    print(f"  InputLayer 설정 수정: batch_shape -> batchInputShape (inputShape 제거)")
                elif 'inputShape' in config and 'batchInputShape' in config:
                    # 둘 다 있으면 inputShape 제거
                    config.pop('inputShape')
                    modified = True
                    print(f"  InputLayer 설정 수정: inputShape 제거 (batchInputShape만 유지)")
        
        if modified:
            with open(model_json_path, 'w') as f:
                json.dump(data, f, indent=None, separators=(',', ':'))
            print("  ✅ model.json의 InputLayer 설정 수정 완료")
        
        return modified
    except Exception as e:
        print(f"  ⚠️  model.json 수정 중 오류: {e}")
        return False


def convert_model(model_path, output_dir):
    """Keras 모델을 TF.js 형식으로 변환"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # tensorflowjs 확인 및 설치
    tfjs = check_and_install_tensorflowjs()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 기존 변환 파일 삭제 (model.json과 .bin 파일)
    print(f"\n기존 변환 파일 정리 중: {output_dir}")
    existing_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.json') or file.endswith('.bin'):
                file_path = os.path.join(output_dir, file)
                os.remove(file_path)
                existing_files.append(file)
    if existing_files:
        print(f"  삭제된 파일: {', '.join(existing_files)}")
    else:
        print("  삭제할 파일 없음")
    
    print(f"\n모델 로드 중: {model_path}")
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    print("모델 로드 완료")
    
    print(f"\n모델 변환 중: {model_path} -> {output_dir}")
    print("이 작업은 몇 분 걸릴 수 있습니다...")
    
    # TF.js로 변환 (모델 객체를 직접 전달)
    try:
        # 최신 API 시도
        tfjs.converters.save_keras_model(model, output_dir)
    except TypeError:
        # 구버전 API 사용
        tfjs.converters.save_keras_model(model, output_dir, quantization_dtype=None)
    
    print(f"\n변환 완료!")
    print(f"출력 디렉토리: {output_dir}")
    
    # 생성된 파일 확인
    generated_files = os.listdir(output_dir)
    print(f"생성된 파일:")
    for file in generated_files:
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"  - {file} ({size:,} bytes)")
    
    # .bin 파일이 생성되었는지 확인
    bin_files = [f for f in generated_files if f.endswith('.bin')]
    if not bin_files:
        print("\n⚠️  경고: .bin 파일이 생성되지 않았습니다!")
        print("  모델이 가중치-only 파일일 수 있습니다.")
        print("  전체 모델(구조+가중치)로 저장되었는지 확인하세요.")
    
    # model.json의 InputLayer 설정 수정 (TF.js 호환성)
    model_json_path = os.path.join(output_dir, 'model.json')
    if os.path.exists(model_json_path):
        print("\nTF.js 호환성 검사 중...")
        fix_inputlayer_in_model_json(model_json_path)
        
        # 가중치와 레이어 이름 일치 확인
        verify_weights_match_layers(model_json_path)
        
        # 수정 후 .bin 파일 확인
        if bin_files:
            print(f"\n✅ 변환 완료: model.json과 {len(bin_files)}개의 .bin 파일이 생성되었습니다.")
        else:
            print("\n❌ .bin 파일이 없습니다. 모델을 다시 확인하세요.")


def main():
    parser = argparse.ArgumentParser(description='Keras 모델을 TF.js 형식으로 변환')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='입력 Keras 모델 파일 (default: model.h5)')
    parser.add_argument('--output', type=str, default='public/model',
                        help='출력 디렉토리 (default: public/model)')
    
    args = parser.parse_args()
    
    convert_model(args.model, args.output)


if __name__ == '__main__':
    main()

