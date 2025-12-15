#!/usr/bin/env python3
"""
Keras 모델을 ONNX 형식으로 변환 (간단한 버전)
"""

import argparse
import os
import sys

def convert_to_onnx(model_path, output_path):
    """Keras 모델을 ONNX 형식으로 변환"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"모델 로드 중: {model_path}")
    
    # TensorFlow/Keras 로드
    import tensorflow as tf
    from tensorflow import keras
    
    model = keras.models.load_model(model_path)
    
    print(f"모델 정보:")
    print(f"  입력 shape: {model.input_shape}")
    print(f"  출력 shape: {model.output_shape}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # ONNX 변환 시도
    print(f"\nONNX 변환 시도 중...")
    try:
        import keras2onnx
        import onnx
        
        print("keras2onnx로 변환 중...")
        onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=13)
        onnx.save_model(onnx_model, output_path)
        
        size = os.path.getsize(output_path)
        print(f"\n✅ ONNX 변환 완료!")
        print(f"📁 출력 파일: {output_path}")
        print(f"📦 파일 크기: {size:,} bytes ({size/1024:.2f} KB)")
        
        # ONNX 모델 검증
        try:
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX 모델 검증 완료")
        except Exception as e:
            print(f"⚠️  ONNX 모델 검증 중 경고: {e}")
            
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        print(f"\n💡 대안: SavedModel 형식을 사용하세요")
        print(f"   TensorFlow.js에서도 SavedModel을 로드할 수 있습니다.")
        return saved_model_path
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Keras 모델을 ONNX 형식으로 변환')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='입력 Keras 모델 파일 (default: model.h5)')
    parser.add_argument('--output', type=str, default='public/model/model.onnx',
                        help='출력 ONNX 파일 경로 (default: public/model/model.onnx)')
    
    args = parser.parse_args()
    
    result = convert_to_onnx(args.model, args.output)
    
    if result.endswith('.onnx'):
        print(f"\n📝 ONNX.js 사용 방법:")
        print(f"   1. npm install onnxruntime-web")
        print(f"   2. 또는 CDN: <script src='https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js'></script>")
        print(f"\n   모델 로드 코드:")
        print(f"   const session = await ort.InferenceSession.create('./model/model.onnx');")
        print(f"   const results = await session.run(feed);")


if __name__ == '__main__':
    main()

