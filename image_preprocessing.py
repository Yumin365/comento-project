
import cv2
import numpy as np
import os
# import pillow
from datasets import load_dataset

# 일부만 가져오기
# food = load_dataset("food101", split="train[:5000]")
# food = food.train_test_split(test_size=0.2)
# food["train"][0]

local_dataset_path = "D:\huggingface\datasets\food101\default\0.0.0\e06acf2a88084f04bce4d4a525165d68e0a36c38"

try:
    # 로컬 경로에서 데이터셋 로드
    # 'data_files' 인자에 .arrow 파일이 있는 경로를 넘겨줍니다.
    # 만약 특정 split(train, validation 등)을 명시하고 싶다면 split='train' 등을 추가
    ds = load_dataset("arrow", data_files=f"{local_dataset_path}*.arrow")

    # 로드된 데이터셋 확인
    print(f"데이터셋 로드 완료! 총 {len(ds['train'])}개의 샘플을 로드했습니다.")
    print("데이터셋 구조:", ds)
    print("첫 번째 샘플의 키:", ds['train'][0].keys())

    # 이미지와 라벨에 접근하여 확인
    first_sample = ds['train'][0]
    image_pil = first_sample['image'] # 일반적으로 'image' 키로 PIL Image 객체가 저장됩니다.
    label = first_sample['label']

    print(f"\n첫 번째 이미지의 라벨: {label}")
    print(f"첫 번째 이미지의 PIL 객체 타입: {type(image_pil)}")
    print(f"첫 번째 이미지 크기: {image_pil.size}")

    # PIL Image를 OpenCV 이미지(NumPy 배열)로 변환하여 처리
    image_np = np.array(image_pil) # PIL Image (RGB)를 NumPy 배열로 변환
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # RGB를 BGR로 변환 (OpenCV 기본)

    # OpenCV 이미지 처리 예시 (예: 회색조 변환)
    gray_image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

    # 변환된 이미지 출력 (선택 사항)
    cv2.imshow("Original Image (BGR)", image_cv2)
    cv2.imshow("Grayscale Image", gray_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"데이터셋 로드 중 오류 발생: {e}")
    print("지정된 'local_dataset_path'가 올바른지, '.arrow' 파일이 해당 경로에 있는지 확인하세요.")
    print("필요한 라이브러리 (datasets, opencv-python, numpy, pillow)가 설치되었는지 확인하세요.")
