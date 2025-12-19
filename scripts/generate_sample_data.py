#!/usr/bin/env python3
"""
KSL 손동작 샘플 데이터 생성 스크립트
각 클래스(0~4)마다 균형있게 랜드마크 데이터 생성
각 클래스가 명확히 구분되도록 개선
"""

import argparse
import numpy as np
import os

def generate_class_landmarks(class_id, num_samples, noise_level=0.03, variation=0.1,
                             rot_deg=10.0, scale_jitter=0.1, shift_xy=0.08,
                             z_loc=-0.1, z_scale=0.08, z_clip=(-0.6, 0.3)):
    """
    특정 클래스의 랜드마크 데이터 생성
    
    MediaPipe Hand Landmarks 구조:
    - 0: 손목 (wrist)
    - 1-4: 엄지 (thumb)
    - 5-8: 검지 (index)
    - 9-12: 중지 (middle)
    - 13-16: 약지 (ring)
    - 17-20: 소지 (pinky)
    
    Args:
        class_id: 클래스 ID (0~4, KSL_1~KSL_5)
        num_samples: 생성할 샘플 수
        noise_level: 노이즈 레벨 (0~1)
        variation: 위치 변화 범위 (0~1)
    
    Returns:
        landmarks: (num_samples, 63) 형태의 랜드마크 데이터
    """
    landmarks = []
    
    for _ in range(num_samples):
        # 기본 랜드마크 패턴 생성 (21개 포인트 × 3차원 = 63차원)
        base_landmarks = np.zeros(63, dtype=np.float32)
        
        # 손목 위치 (항상 동일)
        wrist_x = 0.5 + np.random.uniform(-variation*0.3, variation*0.3)
        wrist_y = 0.8 + np.random.uniform(-variation*0.2, variation*0.2)
        wrist_z = 0.0
        base_landmarks[0:3] = [wrist_x, wrist_y, wrist_z]
        
        # 클래스별 특징 패턴 생성 (각 클래스가 명확히 구분되도록)
        if class_id == 0:  # KSL_1: 검지 하나만 펴기
            # 검지만 위로 펴기 (다른 손가락은 접힘)
            finger_up_y = 0.15 + np.random.uniform(-variation*0.1, variation*0.1)
            
            # 검지 (5-8): 위로 펴기
            base_landmarks[5*3:6*3] = [0.5, 0.5, 0.0]  # 검지 MCP
            base_landmarks[6*3:7*3] = [0.5, 0.35, 0.0]  # 검지 PIP
            base_landmarks[7*3:8*3] = [0.5, 0.25, 0.0]  # 검지 DIP
            base_landmarks[8*3:9*3] = [0.5, finger_up_y, 0.0]  # 검지 끝
            
            # 나머지 손가락은 접힘 (낮은 y값)
            # 중지 (9-12): 접힘
            base_landmarks[9*3:10*3] = [0.5, 0.6, 0.0]
            base_landmarks[10*3:11*3] = [0.5, 0.65, 0.0]
            base_landmarks[11*3:12*3] = [0.5, 0.7, 0.0]
            base_landmarks[12*3:13*3] = [0.5, 0.72, 0.0]
            
            # 약지 (13-16): 접힘
            base_landmarks[13*3:14*3] = [0.55, 0.6, 0.0]
            base_landmarks[14*3:15*3] = [0.55, 0.65, 0.0]
            base_landmarks[15*3:16*3] = [0.55, 0.7, 0.0]
            base_landmarks[16*3:17*3] = [0.55, 0.72, 0.0]
            
            # 소지 (17-20): 접힘
            base_landmarks[17*3:18*3] = [0.6, 0.6, 0.0]
            base_landmarks[18*3:19*3] = [0.6, 0.65, 0.0]
            base_landmarks[19*3:20*3] = [0.6, 0.7, 0.0]
            base_landmarks[20*3:21*3] = [0.6, 0.72, 0.0]
            
            # 엄지 (1-4): 접힘
            base_landmarks[1*3:2*3] = [0.4, 0.65, 0.0]
            base_landmarks[2*3:3*3] = [0.4, 0.7, 0.0]
            base_landmarks[3*3:4*3] = [0.4, 0.72, 0.0]
            base_landmarks[4*3:5*3] = [0.4, 0.74, 0.0]
            
        elif class_id == 1:  # KSL_2: 검지와 중지 펴기
            finger_up_y = 0.15 + np.random.uniform(-variation*0.1, variation*0.1)
            
            # 검지 (5-8): 위로 펴기
            base_landmarks[5*3:6*3] = [0.45, 0.5, 0.0]
            base_landmarks[6*3:7*3] = [0.45, 0.35, 0.0]
            base_landmarks[7*3:8*3] = [0.45, 0.25, 0.0]
            base_landmarks[8*3:9*3] = [0.45, finger_up_y, 0.0]
            
            # 중지 (9-12): 위로 펴기
            base_landmarks[9*3:10*3] = [0.5, 0.5, 0.0]
            base_landmarks[10*3:11*3] = [0.5, 0.35, 0.0]
            base_landmarks[11*3:12*3] = [0.5, 0.25, 0.0]
            base_landmarks[12*3:13*3] = [0.5, finger_up_y, 0.0]
            
            # 나머지 손가락은 접힘
            # 약지 (13-16): 접힘
            base_landmarks[13*3:14*3] = [0.55, 0.6, 0.0]
            base_landmarks[14*3:15*3] = [0.55, 0.65, 0.0]
            base_landmarks[15*3:16*3] = [0.55, 0.7, 0.0]
            base_landmarks[16*3:17*3] = [0.55, 0.72, 0.0]
            
            # 소지 (17-20): 접힘
            base_landmarks[17*3:18*3] = [0.6, 0.6, 0.0]
            base_landmarks[18*3:19*3] = [0.6, 0.65, 0.0]
            base_landmarks[19*3:20*3] = [0.6, 0.7, 0.0]
            base_landmarks[20*3:21*3] = [0.6, 0.72, 0.0]
            
            # 엄지 (1-4): 접힘
            base_landmarks[1*3:2*3] = [0.4, 0.65, 0.0]
            base_landmarks[2*3:3*3] = [0.4, 0.7, 0.0]
            base_landmarks[3*3:4*3] = [0.4, 0.72, 0.0]
            base_landmarks[4*3:5*3] = [0.4, 0.74, 0.0]
            
        elif class_id == 2:  # KSL_3: 검지, 중지, 약지 펴기
            finger_up_y = 0.15 + np.random.uniform(-variation*0.1, variation*0.1)
            
            # 검지 (5-8): 위로 펴기
            base_landmarks[5*3:6*3] = [0.4, 0.5, 0.0]
            base_landmarks[6*3:7*3] = [0.4, 0.35, 0.0]
            base_landmarks[7*3:8*3] = [0.4, 0.25, 0.0]
            base_landmarks[8*3:9*3] = [0.4, finger_up_y, 0.0]
            
            # 중지 (9-12): 위로 펴기
            base_landmarks[9*3:10*3] = [0.5, 0.5, 0.0]
            base_landmarks[10*3:11*3] = [0.5, 0.35, 0.0]
            base_landmarks[11*3:12*3] = [0.5, 0.25, 0.0]
            base_landmarks[12*3:13*3] = [0.5, finger_up_y, 0.0]
            
            # 약지 (13-16): 위로 펴기
            base_landmarks[13*3:14*3] = [0.6, 0.5, 0.0]
            base_landmarks[14*3:15*3] = [0.6, 0.35, 0.0]
            base_landmarks[15*3:16*3] = [0.6, 0.25, 0.0]
            base_landmarks[16*3:17*3] = [0.6, finger_up_y, 0.0]
            
            # 나머지 손가락은 접힘
            # 소지 (17-20): 접힘
            base_landmarks[17*3:18*3] = [0.65, 0.6, 0.0]
            base_landmarks[18*3:19*3] = [0.65, 0.65, 0.0]
            base_landmarks[19*3:20*3] = [0.65, 0.7, 0.0]
            base_landmarks[20*3:21*3] = [0.65, 0.72, 0.0]
            
            # 엄지 (1-4): 접힘
            base_landmarks[1*3:2*3] = [0.35, 0.65, 0.0]
            base_landmarks[2*3:3*3] = [0.35, 0.7, 0.0]
            base_landmarks[3*3:4*3] = [0.35, 0.72, 0.0]
            base_landmarks[4*3:5*3] = [0.35, 0.74, 0.0]
            
        elif class_id == 3:  # KSL_4: 검지, 중지, 약지, 소지 펴기
            finger_up_y = 0.15 + np.random.uniform(-variation*0.1, variation*0.1)
            
            # 검지 (5-8): 위로 펴기
            base_landmarks[5*3:6*3] = [0.35, 0.5, 0.0]
            base_landmarks[6*3:7*3] = [0.35, 0.35, 0.0]
            base_landmarks[7*3:8*3] = [0.35, 0.25, 0.0]
            base_landmarks[8*3:9*3] = [0.35, finger_up_y, 0.0]
            
            # 중지 (9-12): 위로 펴기
            base_landmarks[9*3:10*3] = [0.5, 0.5, 0.0]
            base_landmarks[10*3:11*3] = [0.5, 0.35, 0.0]
            base_landmarks[11*3:12*3] = [0.5, 0.25, 0.0]
            base_landmarks[12*3:13*3] = [0.5, finger_up_y, 0.0]
            
            # 약지 (13-16): 위로 펴기
            base_landmarks[13*3:14*3] = [0.6, 0.5, 0.0]
            base_landmarks[14*3:15*3] = [0.6, 0.35, 0.0]
            base_landmarks[15*3:16*3] = [0.6, 0.25, 0.0]
            base_landmarks[16*3:17*3] = [0.6, finger_up_y, 0.0]
            
            # 소지 (17-20): 위로 펴기
            base_landmarks[17*3:18*3] = [0.65, 0.5, 0.0]
            base_landmarks[18*3:19*3] = [0.65, 0.35, 0.0]
            base_landmarks[19*3:20*3] = [0.65, 0.25, 0.0]
            base_landmarks[20*3:21*3] = [0.65, finger_up_y, 0.0]
            
            # 엄지만 접힘
            base_landmarks[1*3:2*3] = [0.3, 0.65, 0.0]
            base_landmarks[2*3:3*3] = [0.3, 0.7, 0.0]
            base_landmarks[3*3:4*3] = [0.3, 0.72, 0.0]
            base_landmarks[4*3:5*3] = [0.3, 0.74, 0.0]
            
        else:  # class_id == 4: KSL_5: 모든 손가락 펴기
            finger_up_y = 0.15 + np.random.uniform(-variation*0.1, variation*0.1)
            
            # 엄지 (1-4): 위로 펴기
            base_landmarks[1*3:2*3] = [0.3, 0.5, 0.0]
            base_landmarks[2*3:3*3] = [0.3, 0.35, 0.0]
            base_landmarks[3*3:4*3] = [0.3, 0.25, 0.0]
            base_landmarks[4*3:5*3] = [0.3, finger_up_y, 0.0]
            
            # 검지 (5-8): 위로 펴기
            base_landmarks[5*3:6*3] = [0.4, 0.5, 0.0]
            base_landmarks[6*3:7*3] = [0.4, 0.35, 0.0]
            base_landmarks[7*3:8*3] = [0.4, 0.25, 0.0]
            base_landmarks[8*3:9*3] = [0.4, finger_up_y, 0.0]
            
            # 중지 (9-12): 위로 펴기
            base_landmarks[9*3:10*3] = [0.5, 0.5, 0.0]
            base_landmarks[10*3:11*3] = [0.5, 0.35, 0.0]
            base_landmarks[11*3:12*3] = [0.5, 0.25, 0.0]
            base_landmarks[12*3:13*3] = [0.5, finger_up_y, 0.0]
            
            # 약지 (13-16): 위로 펴기
            base_landmarks[13*3:14*3] = [0.6, 0.5, 0.0]
            base_landmarks[14*3:15*3] = [0.6, 0.35, 0.0]
            base_landmarks[15*3:16*3] = [0.6, 0.25, 0.0]
            base_landmarks[16*3:17*3] = [0.6, finger_up_y, 0.0]
            
            # 소지 (17-20): 위로 펴기
            base_landmarks[17*3:18*3] = [0.7, 0.5, 0.0]
            base_landmarks[18*3:19*3] = [0.7, 0.35, 0.0]
            base_landmarks[19*3:20*3] = [0.7, 0.25, 0.0]
            base_landmarks[20*3:21*3] = [0.7, finger_up_y, 0.0]
        
        # 간단한 회전/스케일/이동 증강 (2D 평면상, z는 별도 처리)
        theta = np.deg2rad(np.random.uniform(-rot_deg, rot_deg))
        s = 1.0 + np.random.uniform(-scale_jitter, scale_jitter)
        tx = np.random.uniform(-shift_xy, shift_xy)
        ty = np.random.uniform(-shift_xy, shift_xy)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # 위치 변화 추가 (다양한 각도와 위치)
        for i in range(21):
            idx = i * 3
            x = base_landmarks[idx]
            y = base_landmarks[idx+1]

            # 회전 + 스케일 + 평행이동
            x_rot = x * cos_t - y * sin_t
            y_rot = x * sin_t + y * cos_t
            x_aug = x_rot * s + tx
            y_aug = y_rot * s + ty

            # 랜덤 위치 변화 (작은 jitter)
            x_aug += np.random.uniform(-variation, variation)
            y_aug += np.random.uniform(-variation * 0.5, variation * 0.5)

            # z는 MediaPipe 스타일: 음수 중심 분포를 더 넓게
            z_noise = np.random.normal(loc=z_loc, scale=z_scale)
            z_noise = np.clip(z_noise, z_clip[0], z_clip[1])
            z_aug = base_landmarks[idx + 2] + z_noise

            base_landmarks[idx] = x_aug
            base_landmarks[idx + 1] = y_aug
            base_landmarks[idx + 2] = z_aug
        
        # 노이즈 추가 (변화성 부여) - z도 같이 흔들리지만 큰 범위는 클리핑하지 않음
        noise = np.random.normal(0, noise_level, 63).astype(np.float32)
        sample = base_landmarks + noise
        
        # x,y는 0~1로 클리핑, z는 그대로 유지하여 깊이 분포를 살림
        sample = sample.reshape(21, 3)
        sample[:, 0] = np.clip(sample[:, 0], 0.0, 1.0)  # x
        sample[:, 1] = np.clip(sample[:, 1], 0.0, 1.0)  # y
        # z는 MediaPipe처럼 음수~양수 허용 (추후 추가 정규화 없음)
        sample = sample.reshape(-1)
        
        landmarks.append(sample)
    
    return np.array(landmarks, dtype=np.float32)

def generate_dataset(num_samples_per_class=300, noise_level=0.03, variation=0.1, output_path='data/ksl_landmarks.npz'):
    """
    전체 데이터셋 생성
    
    Args:
        num_samples_per_class: 클래스당 샘플 수 (최소 100개 이상 권장)
        noise_level: 노이즈 레벨
        variation: 위치 변화 범위
        output_path: 출력 파일 경로
    """
    print(f"\n{'='*60}")
    print(f"KSL 손동작 샘플 데이터 생성 (개선 버전)")
    print(f"{'='*60}")
    print(f"클래스당 샘플 수: {num_samples_per_class}")
    print(f"노이즈 레벨: {noise_level}")
    print(f"위치 변화 범위: {variation}")
    print(f"총 샘플 수: {num_samples_per_class * 5}")
    
    X_list = []
    y_list = []
    
    class_names = ['KSL_1', 'KSL_2', 'KSL_3', 'KSL_4', 'KSL_5']
    
    for class_id in range(5):
        print(f"\n클래스 {class_id} ({class_names[class_id]}) 생성 중...")
        landmarks = generate_class_landmarks(class_id, num_samples_per_class, noise_level, variation)
        X_list.append(landmarks)
        y_list.append(np.full(num_samples_per_class, class_id, dtype=np.int32))
        print(f"  생성 완료: {landmarks.shape[0]}개 샘플")
        print(f"  샘플 범위: X=[{landmarks.min():.3f}, {landmarks.max():.3f}]")
    
    # 데이터 결합
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # 데이터 셔플
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"\n{'='*60}")
    print(f"데이터셋 생성 완료")
    print(f"{'='*60}")
    print(f"X 형태: {X.shape}")
    print(f"y 형태: {y.shape}")
    print(f"X 범위: [{X.min():.4f}, {X.max():.4f}]")
    
    # 클래스 분포 확인
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\n클래스 분포:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(y)) * 100
        print(f"  클래스 {cls} ({class_names[cls]}): {count}개 ({percentage:.1f}%)")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 데이터 저장
    np.savez_compressed(output_path, X=X, y=y)
    print(f"\n✅ 데이터 저장 완료: {output_path}")
    
    file_size = os.path.getsize(output_path)
    print(f"파일 크기: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

def main():
    parser = argparse.ArgumentParser(description='KSL 손동작 샘플 데이터 생성 (개선 버전)')
    parser.add_argument('--samples-per-class', type=int, default=300,
                        help='클래스당 샘플 수 (default: 300, 최소 100 권장)')
    parser.add_argument('--noise-level', type=float, default=0.03,
                        help='노이즈 레벨 (default: 0.03)')
    parser.add_argument('--variation', type=float, default=0.1,
                        help='위치 변화 범위 (default: 0.1)')
    parser.add_argument('--output', type=str, default='data/ksl_landmarks.npz',
                        help='출력 파일 경로 (default: data/ksl_landmarks.npz)')
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정 (재현성)
    np.random.seed(42)
    
    generate_dataset(
        num_samples_per_class=args.samples_per_class,
        noise_level=args.noise_level,
        variation=args.variation,
        output_path=args.output
    )
    
    print(f"\n✅ 샘플 데이터 생성 완료!")
    print(f"\n다음 단계: 모델 학습")
    print(f"  python3 scripts/train_ksl.py --data {args.output} --epochs 100")

if __name__ == '__main__':
    main()
