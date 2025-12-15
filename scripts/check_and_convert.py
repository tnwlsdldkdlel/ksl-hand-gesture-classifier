#!/usr/bin/env python3
"""
tensorflowjs 설치 상태 확인 및 자동 변환 스크립트
"""

import subprocess
import sys
import time
import os

def check_tensorflowjs_installed():
    """tensorflowjs 설치 여부 확인"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "tensorflowjs"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def wait_for_installation(max_wait_minutes=20):
    """설치 완료까지 대기"""
    print("tensorflowjs 설치 완료를 기다리는 중...")
    print(f"최대 대기 시간: {max_wait_minutes}분")
    
    start_time = time.time()
    check_interval = 30  # 30초마다 확인
    
    while True:
        elapsed_minutes = (time.time() - start_time) / 60
        
        if check_tensorflowjs_installed():
            print(f"\n✅ tensorflowjs 설치 완료! (소요 시간: {elapsed_minutes:.1f}분)")
            return True
        
        if elapsed_minutes >= max_wait_minutes:
            print(f"\n⏰ 최대 대기 시간({max_wait_minutes}분) 초과")
            return False
        
        print(f"  대기 중... ({elapsed_minutes:.1f}분 경과)", end='\r')
        time.sleep(check_interval)

def convert_model(model_path, output_dir):
    """모델 변환"""
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    try:
        import tensorflowjs as tfjs
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n🔄 모델 변환 중: {model_path} -> {output_dir}")
        
        tfjs.converters.save_keras_model(
            model_path,
            output_dir,
            quantization_dtype=None
        )
        
        print(f"\n✅ 변환 완료!")
        print(f"📁 출력 디렉토리: {output_dir}")
        print(f"\n생성된 파일:")
        for file in sorted(os.listdir(output_dir)):
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")
        
        return True
        
    except ImportError:
        print("❌ tensorflowjs를 import할 수 없습니다.")
        return False
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='tensorflowjs 설치 확인 및 모델 변환')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='입력 모델 파일 (default: model.h5)')
    parser.add_argument('--output', type=str, default='public/model',
                        help='출력 디렉토리 (default: public/model)')
    parser.add_argument('--wait', action='store_true',
                        help='설치 완료까지 대기')
    parser.add_argument('--max-wait', type=int, default=20,
                        help='최대 대기 시간(분) (default: 20)')
    
    args = parser.parse_args()
    
    # 설치 확인
    if not check_tensorflowjs_installed():
        if args.wait:
            if not wait_for_installation(args.max_wait):
                print("\n설치가 완료되지 않았습니다. 수동으로 설치해주세요:")
                print("  pip3 install tensorflowjs")
                sys.exit(1)
        else:
            print("❌ tensorflowjs가 설치되지 않았습니다.")
            print("설치 중이면 --wait 옵션을 사용하거나, 설치 완료 후 다시 실행하세요.")
            sys.exit(1)
    
    # 변환 실행
    success = convert_model(args.model, args.output)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

