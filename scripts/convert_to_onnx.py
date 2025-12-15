#!/usr/bin/env python3
"""
Keras 모델을 ONNX 형식으로 변환하는 스크립트
"""

import argparse
import os
import sys

try:
    import tensorflow as tf
    from tensorflow import keras
    import onnx
    try:
        import keras2onnx
    except ImportError:
        import tf2onnx
        keras2onnx = None
except ImportError as e:
    print(f"필요한 패키지가 설치되지 않았습니다: {e}")
    print("\n설치 명령어:")
    print("  pip3 install tensorflow onnx keras2onnx")
    sys.exit(1)


def convert_to_onnx(model_path, output_path, opset=13):
    """Keras 모델을 ONNX 형식으로 변환"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"모델 로드 중: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"모델 정보:")
    print(f"  입력 shape: {model.input_shape}")
    print(f"  출력 shape: {model.output_shape}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nONNX 변환 중: {model_path} -> {output_path}")
    print(f"  Opset 버전: {opset}")
    
    # ONNX로 변환
    if keras2onnx:
        # keras2onnx 사용
        onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=opset)
        onnx.save_model(onnx_model, output_path)
    else:
        # tf2onnx 사용
        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        output_path_onnx = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset,
            output_path=output_path
        )
    
    # 파일 크기 확인
    size = os.path.getsize(output_path)
    print(f"\n✅ 변환 완료!")
    print(f"📁 출력 파일: {output_path}")
    print(f"📦 파일 크기: {size:,} bytes ({size/1024:.2f} KB)")
    
    # ONNX 모델 검증
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 모델 검증 완료")
    except Exception as e:
        print(f"⚠️  ONNX 모델 검증 중 경고: {e}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Keras 모델을 ONNX 형식으로 변환')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='입력 Keras 모델 파일 (default: model.h5)')
    parser.add_argument('--output', type=str, default='public/model/model.onnx',
                        help='출력 ONNX 파일 경로 (default: public/model/model.onnx)')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset 버전 (default: 13)')
    
    args = parser.parse_args()
    
    convert_to_onnx(args.model, args.output, args.opset)
    
    print(f"\n📝 ONNX.js 사용 방법:")
    print(f"   npm install onnxruntime-web")
    print(f"   또는")
    print(f"   <script src='https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js'></script>")
    print(f"\n   모델 로드:")
    print(f"   const session = await ort.InferenceSession.create('./model/model.onnx');")


if __name__ == '__main__':
    main()

