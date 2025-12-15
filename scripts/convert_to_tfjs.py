#!/usr/bin/env python3
"""
Keras 모델을 TF.js 형식으로 변환하는 스크립트
"""

import argparse
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


def convert_model(model_path, output_dir):
    """Keras 모델을 TF.js 형식으로 변환"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # tensorflowjs 확인 및 설치
    tfjs = check_and_install_tensorflowjs()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
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
    print(f"생성된 파일:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"  - {file} ({size:,} bytes)")


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

